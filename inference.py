from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import joblib
import argparse
from pathlib import Path
import requests
import fitz
import os
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

model_filename = 'model/log_regression_CountVectorizer.pkl' 
classifier = joblib.load(model_filename)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

vectorizer = CountVectorizer()

train_df = pd.read_pickle('data/train_preprocessed.pickle')
vectorizer.fit(train_df['cleaned_text'])

def pdf2text(url):
    pdf_file_name = Path(url).name

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  
        with open(pdf_file_name, "wb") as file:
            file.write(response.content)

        doc = fitz.open(pdf_file_name)
        text = ""
        for page in doc:
            text += page.get_text()

        return text
    except Exception as err:
        print(url + " " + str(err))
        return ""
    finally:
        if os.path.exists(pdf_file_name):
            os.remove(pdf_file_name)

def preprocess_text(text):
    text = text.lower()    
    text = text.replace('\t', ' ').replace('\n', ' ')
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Main function
def predict_class(url):
    text = pdf2text(url)
    cleaned_text = preprocess_text(text)
    X = vectorizer.transform([cleaned_text])
    prediction = classifier.predict(X)
    probability = classifier.predict_proba(X)
    return prediction[0], probability.max()

parser = argparse.ArgumentParser(description='Predict the class of a PDF document.')
parser.add_argument('url', type=str, help='The path to the PDF file')
args = parser.parse_args()
url = args.url

predicted_class, predicted_probability = predict_class(url)
print(f'Predicted Class: {predicted_class}')
print(f'Predicted Probability: {predicted_probability:.2f}')
