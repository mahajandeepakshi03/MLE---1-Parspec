# Steps : 

1. git clone <repo_link>
2. cd classification
3. pip install -r requirements.txt
4. python3 <script_name>.py


# This repo contains 4 scripts : 

1. data_extraction.py : To extract the data from the url of the given pdf. To execute it : 

    Run : python3 data_extraction.py

2. preprocessing.py : To clean the text extracted from the pdf. To execute it : 

    Run : python3 preprocessing.py

3. train.py : Here, the vectorization pf the cleaned text is done. After that, training is done on multiple classification models. To execute it : 

    Run : python3 train.py

4. inference.py : This is the inference pipeline. To execute it : 

    Run : python3 inference.py <url_of_the_pdf>