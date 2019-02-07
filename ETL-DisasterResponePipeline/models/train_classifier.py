import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
import re
nltk.download(['wordnet', 'punkt', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

################
# Imported from https://github.com/udacity/workspaces-student-support/tree/master/jupyter here, as 
# my workspace kept closing, before finishing training
# Snippet begins below the comment
################

import signal
 
from contextlib import contextmanager
 
import requests
 
 
DELAY = INTERVAL = 4 * 60  # interval time in seconds
MIN_DELAY = MIN_INTERVAL = 2 * 60
KEEPALIVE_URL = "https://nebula.udacity.com/api/v1/remote/keep-alive"
TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token"
TOKEN_HEADERS = {"Metadata-Flavor":"Google"}
 
 
def _request_handler(headers):
    def _handler(signum, frame):
        requests.request("POST", KEEPALIVE_URL, headers=headers)
    return _handler
 
 
@contextmanager
def active_session(delay=DELAY, interval=INTERVAL):
    """
    Example:
 
    from workspace_utils import active_session
 
    with active_session():
        # do long-running work here
    """
    token = requests.request("GET", TOKEN_URL, headers=TOKEN_HEADERS).text
    headers = {'Authorization': "STAR " + token}
    delay = max(delay, MIN_DELAY)
    interval = max(interval, MIN_INTERVAL)
    original_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _request_handler(headers))
        signal.setitimer(signal.ITIMER_REAL, delay, interval)
        yield
    finally:
        signal.signal(signal.SIGALRM, original_handler)
        signal.setitimer(signal.ITIMER_REAL, 0)
 
 
def keep_awake(iterable, delay=DELAY, interval=INTERVAL):
    """
    Example:
 
    from workspace_utils import keep_awake
 
    for i in keep_awake(range(5)):
        # do iteration with lots of work here
    """
    with active_session(delay, interval): yield from iterable

#############
# End of snippet
#############

def load_data(database_filepath):
    """
    load data from database
    :param database_filepath: Takes the filepath to the database it should load.
    :return: Returns X, y and a list of categories.
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponseData', engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    categories = y.columns.values
    return X, y, categories

# Created new tokenize function
def tokenize(text):
    """
    Function taken from the run.py file provided in the workspace
    :param text: Takes in text
    :return: list of tokens
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    words = word_tokenize(text)
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]

    return lemmed


def build_model():
    """
    Builds the model using pipeline and gridserach, for optimizing
    :return:
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    # Parameters for GridSearch (simplified due to time challenges :-( )
    parameters = {
        'vect__ngram_range': [(1, 1),(2,2)],
        'vect__max_features': [2,5,10],
        'clf__estimator__n_estimators': [25,40,70]}

    cv = GridSearchCV(pipeline, param_grid=parameters, cv=5, verbose=10)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    classification_report(y_test, y_pred, target_names=categories)
    :param model: Takes in a fitted model
    :param X_test: Takes in test data X value
    :param Y_test: Takes in test data Y value
    :param category_names: Takes in a list of category names
    :return: Nothing, prints out simply a classification report.
    """
    y_pred = model.predict(X_test)
    # The commented out code was the old solution, got an error where I used target_name, instead of target_names
    # print(classification_report(Y_test, y_pred, target_names=category_names))
    print("-------Classification Report-------\n")
    for i in range(len(category_names)):
        print("Label:", category_names[i], '\n')
        print(classification_report(Y_test.loc[:, category_names[i]], y_pred[:, i]))
        print(11*'-----', '\n')

def save_model(model, model_filepath):
    """
    Saves the model
    :param model: Takes in the model, which should be saved
    :param model_filepath: Takes in the filepath, which acts as the name and location for the save.
    :return: Nothing, as it just saves the file.
    """
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
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)   
        print('Trained model saved!')
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    # The following addition is necessary in order for the udacity workstation to stay awake while training.
    with active_session():
    # do long-running work here
        main()