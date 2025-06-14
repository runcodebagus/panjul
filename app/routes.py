from flask import Flask, render_template, request, redirect, url_for
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


# Load the model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    count_vectorizer = pickle.load(vectorizer_file)

nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))
translator = str.maketrans('', '', string.punctuation)

def preprocess_text(text):
    text = text.lower()
    text = text.translate(translator)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

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
    if request.method == 'POST':
        texts = request.form.getlist('text')  # Mengambil daftar teks dari formulir
        preprocessed_texts = [preprocess_text(text) for text in texts]
        
        # Menggunakan Count Vectorizer untuk mentransformasi teks-teks ini
        text_vectors = count_vectorizer.transform(preprocessed_texts)
        
        sentiments = [model.predict(vector)[0] for vector in text_vectors]

        results = zip(texts, sentiments)
        # Kembali ke halaman hasil
        return render_template('analisis.html', results=results)



@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files.get('file')

        if uploaded_file and uploaded_file.filename != '':
            filename = secure_filename(uploaded_file.filename)
            file_extension = os.path.splitext(filename)[1].lower()

            if file_extension in ['.csv', '.xls', '.xlsx']:
                try:
                    # Baca file sesuai format
                    if file_extension == '.csv':
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)

                    # Cek kolom 'Komentar' ada atau tidak
                    if 'Komentar' not in df.columns:
                        return render_template('upload.html', error_message="Kolom 'Komentar' tidak ditemukan di file.")

                    # Preprocessing & prediksi
                    df['Preprocessed Text'] = df['Komentar'].apply(preprocess_text)
                    text_vectors = count_vectorizer.transform(df['Preprocessed Text'])
                    df['Sentiment'] = model.predict(text_vectors)
                    df['Sentiment'] = df['Sentiment'].map({0: 'bulliying', 1: 'non-bulliying'})

                    # Simpan hasil ke static/results.csv agar bisa didownload
                    output_path = os.path.join('static', 'results.csv')
                    df.to_csv(output_path, index=False)

                    return render_template(
                        'upload.html',
                        prediction_success_message='Analisis berhasil! Hasil ditampilkan di bawah.',
                        df=df
                    )
                except Exception as e:
                    return render_template('upload.html', error_message=f'Terjadi kesalahan saat memproses file: {str(e)}')
            else:
                return render_template('upload.html', error_message='Hanya file .csv, .xls, dan .xlsx yang diperbolehkan.')
        else:
            return render_template('upload.html', error_message='Silakan pilih file untuk diunggah.')

    return render_template('upload.html')
