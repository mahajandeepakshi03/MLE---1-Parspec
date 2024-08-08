from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

train_df = pd.read_pickle('data/train_preprocessed.pickle')
test_df = pd.read_pickle('data/test_preprocessed.pickle')
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

vectorizers = {
    'TfidfVectorizer': TfidfVectorizer(),
    'CountVectorizer': CountVectorizer(),
}

classifiers = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'NaiveBayes': MultinomialNB()
}

for vec_name, vectorizer in vectorizers.items():
    print(f"*********************************{vectorizer}*****************************************************")
    X_train = vectorizer.fit_transform(train_df['cleaned_text'])
    X_test = vectorizer.transform(test_df['cleaned_text'])
    
    for clf_name, classifier in classifiers.items():
        print(f"********{clf_name}*********")
        classifier.fit(X_train, train_df['target_col'])
        model_filename = f'model/{clf_name}_{vec_name}.pkl'
        joblib.dump(classifier, model_filename)
        y_pred = classifier.predict(X_test)

        accuracy = accuracy_score(test_df['target_col'], y_pred)
        precision = precision_score(test_df['target_col'], y_pred, average='weighted')
        recall = recall_score(test_df['target_col'], y_pred, average='weighted')
        f1 = f1_score(test_df['target_col'], y_pred, average='weighted')

        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")

        # Save results to Excel
        results_df = pd.DataFrame({
            'Predicted': y_pred,
            'Actual': test_df['target_col']
        })
        results_filename = f'results/predictions_vs_actual_{clf_name}_{vec_name}.xlsx'
        results_df.to_excel(results_filename, index=False)

