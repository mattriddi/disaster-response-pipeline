import sys
import pandas as pd
import numpy as np
import sqlite3
import re
import nltk
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download(['stopwords', 'punkt', 'wordnet'])


def load_data(database_filepath):
    """
    This function loads the data and splits it into the input variable and the
    target variable
    
    INPUT: The filepath for the database
    
    OUTPUT: The input variable and the target variable for the machine learning
    model in 2 dataframes - X and Y    
    """
    
    engine = create_engine(('sqlite:///' + database_filepath))
    df = pd.read_sql("SELECT * FROM disaster_response_table" , engine)
    X = df['message']
    Y = df.iloc[:, -36:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """
    This function normalises case, lemmatises and tokenises text
    
    INPUT: Text as strings
    
    OUTPUT: Case normalised, lemmatised and tokenised text
    """
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    This function returns a multi-output machine learning model pipeline and uses
    GridSearchCV to optimise hyperparameters for this model    
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators' : [5, 10],
        'clf__estimator__max_depth' : [None, 10]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function evaluates the model performance by predicting using the test data
    set and comparting this to the ground truth
    
    INPUT: The model, the input and target variable from the test dataset and the
    category names
    
    OUTPUT: Prints the model F1 score, precision and recall for each category
    """
    y_pred = model.predict(X_test)
    i = 0
    while i<Y_test.shape[1]:
        print(category_names[i], ': \n', classification_report(Y_test.iloc[:, i], y_pred[:, i]))
        i = i + 1



def save_model(model, model_filepath):
    """
    This function exports the model to the specified model filepath as a pickle file
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    """
    This function loads the data then builds, trains and evaluates the model, which is then
    saved as a pickle file
    """
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()