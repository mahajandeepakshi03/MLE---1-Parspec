import pandas as pd
import fitz
import requests
from pathlib import Path
import os

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

def process_batch(data, start_index, batch_size):
    for idx, row in data.iloc[start_index:start_index+batch_size].iterrows():
        print(idx, row['datasheet_link'])
        data.at[idx, 'text'] = pdf2text(row['datasheet_link'])

    return data

batch_size = 100
start_index = 0

train_data = pd.read_csv('data/train_data.csv')
train_data.drop_duplicates(inplace=True)
train_data.reset_index(inplace=True)

while start_index < len(train_data):
    train_data = process_batch(train_data, start_index, batch_size)
    start_index += batch_size
    train_data.to_pickle(f'train_data_batch_{start_index // batch_size}.pickle')

train_data.to_pickle('data/train_data.pickle')


batch_size = 100
start_index = 0

test_data = pd.read_csv('data/test_data.csv')
test_data.drop_duplicates(inplace=True)
test_data.reset_index(inplace=True)

while start_index < len(test_data):
    test_data = process_batch(test_data, start_index, batch_size)
    start_index += batch_size
    test_data.to_pickle(f'test_data_batch_{start_index // batch_size}.pickle')

test_data.to_pickle('data/test_data.pickle')
