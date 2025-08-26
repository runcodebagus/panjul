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
import openpyxl  # pastikan terpasang untuk export Excel
from werkzeug.utils import secure_filename

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

# ====== SIMPLE IN-MEMORY CACHE (OPSI A) ======
from flask import send_file, flash  # send_file sudah diimpor, flash dipakai di download
import io, uuid, time

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
    # bulk predict dari textarea/field multiple
    texts = request.form.getlist('text')  # daftar teks
    preprocessed_texts = [preprocess_text(t) for t in texts]
    text_vectors = count_vectorizer.transform(preprocessed_texts)
    sentiments = model.predict(text_vectors)  # hasil numerik sesuai training
    # kalau mau tampil string, siapkan mapping di sini juga
    label_map = {0: 'bullying', 1: 'non-bullying'}
    # gunakan Series agar sejajar & mudah dipetakan
    s_pred = pd.Series(sentiments).replace(label_map)
    results = list(zip(texts, s_pred))
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

            # --- pastikan kolom 'Komentar' ada (fallback beberapa nama umum) ---
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

            # --- prediksi (bulk) ---
            y_pred = model.predict(text_vectors)

            # --- mapping label AMAN (perbaikan error ndarray) ---
            label_map = {0: 'bullying', 1: 'non-bullying'}
            # Opsi aman 1: jadikan Series lalu replace
            s_pred = pd.Series(y_pred, index=df.index)
            df['Sentiment'] = s_pred.replace(label_map)

            # --- pilih kolom yang akan ditampilkan/diunduh ---
            result_df = df[['Komentar', 'Sentiment']].copy()

            # --- simpan ke cache, kirim token ke template ---
            token = _put_cache(result_df)

            return render_template(
                'upload.html',
                prediction_success_message='Analisis berhasil! Hasil ditampilkan di bawah.',
                df=result_df,
                token=token
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
