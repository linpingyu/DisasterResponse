import sys
import pickle

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sqlalchemy import create_engine
import re

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])


def load_data(database_filepath):
    '''
    Loads data from database
    Returns:
        X: feature
        Y: target labels
        category_names: target categories
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disasterCleaned', con=engine)
    X = df.message
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):

    '''
    Tokenizes the text after removing punctuation, and stopwords
    Returns: tokens from the text
    '''

    #init lemmatizer
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    # normalize case and remove punctuation
    text = re.sub(r'[^0-9A-Za-z]', ' ', text.lower())
    #tokenize text
    tokens = word_tokenize(text)
    #lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

# class StartingVerbExtractor(BaseEstimator, TransformerMixin):
#     '''
#     Extracts if the first tag is a verb
#     '''

#     def starting_verb(self, text):
#         sentence_list = nltk.sent_tokenize(text)
#         for sentence in sentence_list:
#             pos_tags = nltk.pos_tag(tokenize(sentence))
#             try:
#                 first_word, first_tag = pos_tags[0]
#                 if first_tag in ['VB', 'VBP'] or first_word == 'RT':
#                     return True
#             except:
#                 continue
#         return False

#     def fit(self, x, y=None):
#         return self

#     def transform(self, X):
#         X_tagged = pd.Series(X).apply(self.starting_verb)
#         return pd.DataFrame(X_tagged)
    


def build_model():
    '''
    Builds classification model pipeline using randomforest classifier
    '''
    pipeline = Pipeline([
        ('features', 
         FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer = tokenize)),
                ('tfidf', TfidfTransformer())
            ])
            )
         ])
        ),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])

    #grid search parameters
    parameters = {
        # 'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        # 'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        # 'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        # 'features__text_pipeline__tfidf__use_idf': (True, False)
    }

    #gridsearch crossvalidations
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=2, n_jobs=4, cv=2)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates the model on the test dataset
    '''

    y_preds = model.predict(X_test)
    for i, column in enumerate(category_names):
        y_pred = [x[i] for x in y_preds]
        print(f'For category: {column}')
        print(classification_report(y_true = Y_test[column], y_pred = y_pred))



def save_model(model, model_filepath):
    '''
    Saves the model into a Python pickle
    '''

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
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
