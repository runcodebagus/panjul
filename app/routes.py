from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import os
from app import app
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import openpyxl  
from werkzeug.utils import secure_filename
import io, uuid, time
from collections import Counter
import re



# ====== LOAD MODEL & VECTORIZER ======
with open('model/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('model/vectorizer.pkl', 'rb') as vectorizer_file:
    count_vectorizer = pickle.load(vectorizer_file)

# ====== NLTK STOPWORDS & PREPROCESS ======
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))
translator = str.maketrans('', '', string.punctuation)

def preprocess_text(text):
    text = str(text).lower()
    text = text.translate(translator)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def batch_top_unigram(preprocessed_series: pd.Series, min_len: int = 3, top_n: int = 10):
    c = Counter()
    for s in preprocessed_series.dropna().astype(str):
        toks = [w for w in s.split() if len(w) >= min_len]
        c.update(toks)
    return c.most_common(top_n)

CACHE_TTL_SECONDS = 60 * 30  # 30 menit
# token -> {"df": DataFrame, "exp": epoch_seconds}
RESULT_CACHE = {}

def _put_cache(df):
    """Simpan DataFrame ke cache dengan TTL, kembalikan token UUID."""
    token = str(uuid.uuid4())
    RESULT_CACHE[token] = {"df": df, "exp": time.time() + CACHE_TTL_SECONDS}
    return token

def _get_cache(token):
    """Ambil DataFrame dari cache (cek kadaluarsa)."""
    item = RESULT_CACHE.get(token)
    if not item:
        return None
    if time.time() > item["exp"]:
        RESULT_CACHE.pop(token, None)
        return None
    return item["df"]

# ====== ROUTES ======
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/analisis')
def analisis():
    return render_template('analisis.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    texts = request.form.getlist('text')  # daftar teks
    preprocessed_texts = [preprocess_text(t) for t in texts]
    text_vectors = count_vectorizer.transform(preprocessed_texts)
    sentiments = model.predict(text_vectors)

    label_map = {0: 'bullying', 1: 'non-bullying'}
    s_pred = pd.Series(sentiments).replace(label_map)
    results = list(zip(texts, s_pred))

    # tidak ada simpan CSV, tidak ada hitung/top
    return render_template('analisis.html', results=results)

def analyze_sentiment():
    texts = request.form.getlist('text')  # daftar teks
    preprocessed_texts = [preprocess_text(t) for t in texts]
    text_vectors = count_vectorizer.transform(preprocessed_texts)
    sentiments = model.predict(text_vectors)

    label_map = {0: 'bullying', 1: 'non-bullying'}
    s_pred = pd.Series(sentiments).replace(label_map)
    results = list(zip(texts, s_pred))

    # === SIMPAN KE CSV ===
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    new_file = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "original_text", "sentiment"])
        ts = datetime.utcnow().isoformat()
        for text, label in results:
            writer.writerow([ts, text, label])

    # === HITUNG TOP PENCARIAN DARI CSV ===
    top_results = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            texts_all = [row["original_text"].strip() for row in reader if row.get("original_text")]
            counts = Counter(texts_all)
            top_results = counts.most_common(10)  # Top 10

    # kirim hasil prediksi & top search ke template
    return render_template('analisis.html', results=results, top_results=top_results)
    texts = request.form.getlist('text')  # daftar teks
    preprocessed_texts = [preprocess_text(t) for t in texts]
    text_vectors = count_vectorizer.transform(preprocessed_texts)
    sentiments = model.predict(text_vectors)  # hasil numerik sesuai training

    label_map = {0: 'bullying', 1: 'non-bullying'}
    s_pred = pd.Series(sentiments).replace(label_map)
    results = list(zip(texts, s_pred))

    # === SIMPAN KE CSV ===
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    new_file = not os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "original_text", "sentiment"])
        ts = datetime.utcnow().isoformat()
        for text, label in results:
            writer.writerow([ts, text, label])

    return render_template('analisis.html', results=results)

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files.get('file')

        if not uploaded_file or uploaded_file.filename == '':
            return render_template('upload.html', error_message='Silakan pilih file untuk diunggah.')

        filename = secure_filename(uploaded_file.filename)
        file_extension = os.path.splitext(filename)[1].lower()

        if file_extension not in ['.csv', '.xls', '.xlsx']:
            return render_template('upload.html', error_message='Hanya file .csv, .xls, dan .xlsx yang diperbolehkan.')

        try:
            # --- baca file ---
            if file_extension == '.csv':
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # --- pastikan kolom 'Komentar' ada ---
            if 'Komentar' not in df.columns:
                for alt in ['komentar', 'text', 'tweet', 'ulasan']:
                    if alt in df.columns:
                        df.rename(columns={alt: 'Komentar'}, inplace=True)
                        break
            if 'Komentar' not in df.columns:
                return render_template('upload.html', error_message="Kolom 'Komentar' tidak ditemukan di file.")

            # --- preprocess & vektorisasi ---
            df['Komentar'] = df['Komentar'].astype(str)
            df['Preprocessed Text'] = df['Komentar'].apply(preprocess_text)
            text_vectors = count_vectorizer.transform(df['Preprocessed Text'])

            # --- prediksi ---
            y_pred = model.predict(text_vectors)
            label_map = {0: 'bullying', 1: 'non-bullying'}
            s_pred = pd.Series(y_pred, index=df.index)
            df['Sentiment'] = s_pred.replace(label_map)

            # --- hasil untuk ditampilkan/unduh ---
            result_df = df[['Komentar', 'Sentiment']].copy()
            token = _put_cache(result_df)  # kalau mau tombol download tetap ada

            # === HITUNG Top kata dari batch upload INI saja ===
            top_unigram = batch_top_unigram(df['Preprocessed Text'], min_len=3, top_n=10)

            return render_template(
                'upload.html',
                prediction_success_message='Analisis berhasil! Hasil ditampilkan di bawah.',
                uploaded_filename=filename,
                df=result_df,
                token=token,
                top_unigram=top_unigram
            )

        except Exception as e:
            return render_template('upload.html', error_message=f'Terjadi kesalahan saat memproses file: {str(e)}')

    # GET
    return render_template('upload.html')


@app.route('/download_results/<token>')
def download_results(token):
    fmt = request.args.get('format', 'xlsx').lower()
    df = _get_cache(token)

    if df is None or df.empty:
        flash('Hasil tidak ditemukan atau sudah kedaluwarsa.', 'danger')
        return redirect(url_for('upload'))

    if fmt == 'csv':
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        data = io.BytesIO(buf.getvalue().encode('utf-8-sig'))
        return send_file(
            data,
            mimetype='text/csv',
            as_attachment=True,
            download_name='hasil_prediksi.csv'
        )

    # default: xlsx
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Hasil')
    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='hasil_prediksi.xlsx'
    )
