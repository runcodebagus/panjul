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
    return render_template('base.html')

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
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            file_extension = os.path.splitext(uploaded_file.filename)[1]
            if file_extension.lower() in ['.csv', '.xls', '.xlsx']:
                if file_extension.lower() == '.csv':
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Preprocess and analyze sentiment for each row
                df['Preprocessed Text'] = df['Komentar'].apply(preprocess_text)
                text_vectors = count_vectorizer.transform(df['Preprocessed Text'])  # Menggunakan Count Vectorizer
                df['Sentiment'] = model.predict(text_vectors)
                df['Sentiment'] = df['Sentiment'].map({0: 'bulliying', 1: 'non-bulliying'})
                
                # Save the results
                output_filename = 'results.csv'
                df.to_csv(output_filename, index=False)
                return redirect(url_for('upload', uploaded=True))
            else:
                return render_template('upload.html', error_message='Only CSV and Excel files are supported.')
    
    return render_template('upload.html')
