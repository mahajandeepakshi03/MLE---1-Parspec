import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

train_df = pd.read_pickle('data/train_data.pickle')
test_df = pd.read_pickle('data/test_data.pickle')

def preprocess_text(text):
    text = text.lower()    
    text = text.replace('\t', ' ').replace('\n', ' ')
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

train_df['cleaned_text'] = train_df['text'].apply(preprocess_text)
train_df2 = train_df[train_df['text'].str.strip() != '']
train_df2 = train_df2[['datasheet_link', 'target_col', 'cleaned_text']]
train_df2.to_pickle('data/train_preprocessed.pickle')

test_df['cleaned_text'] = test_df['text'].apply(preprocess_text)
test_df2 = test_df[test_df['text'].str.strip() != '']
test_df2 = test_df2[['datasheet_link', 'target_col', 'cleaned_text']]
test_df2.to_pickle('data/test_preprocessed.pickle')