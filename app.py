from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords

app = Flask(__name__)

# Load the sentiment model and vectorizer
with open("sentiment_model.pickle", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pickle", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Inisialisasi stopwords bahasa Indonesia
stop_words = set(stopwords.words("indonesian"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form.get("text")
    if text:
        processed_text = preprocess_text(text)
        vectorized_text = vectorizer.transform([processed_text])
        sentiment = model.predict(vectorized_text)[0]
        return render_template("analyze.html", text=text, sentiment=sentiment)
    return render_template("index.html", error="Please provide a text.")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            df = pd.read_csv(file)
            df["Sentiment"] = df["Text"].apply(classify_sentiment)
            return render_template("upload.html", df=df.to_html(classes="table table-striped"))
    return render_template("upload.html")

def preprocess_text(text):
    words = nltk.word_tokenize(text)  # Tokenize kata-kata
    words = [word.lower() for word in words if word.isalpha()]  # Hilangkan karakter non-alfabet
    words = [word for word in words if word not in stop_words]  # Hilangkan stopwords
    preprocessed_text = " ".join(words)  # Gabungkan kata-kata kembali menjadi teks
    return preprocessed_text

def classify_sentiment(text):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    sentiment = model.predict(vectorized_text)[0]
    return sentiment

if __name__ == "__main__":
    app.run(debug=True)
