import string
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# Read CSV
messages = pd.read_csv('spam.csv', encoding='ISO-8859-1')
messages = messages.rename(columns={'v1': 'Label', 'v2': 'Message'})  # renaming columns

# Text Processing
def process_text(message):
    """
    This function performs tokenization of input string
    1. Remove punctuations
    2. Remove stopwords
    3. Returns the list of clean text words
    """
    no_punctuation = [char for char in message if char not in string.punctuation]
    no_punctuation = ''.join(no_punctuation)   # Join all the characters in the array
    tokenized_message = [word for word in no_punctuation.split() if word.lower() not in stopwords.words('english')]
    return tokenized_message

# Create Pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=process_text)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),                    # integer counts to weighted TF-IDF scores
    ('classifier', MLPClassifier())                   # train on TF-IDF vectors with MLP classifier
])

# Fitting using pipeline
pipeline.fit(messages['Message'], messages['Label'])

# Input from user
input_message = input("Enter the message: ")

# Predicting using pipeline
prediction = pipeline.predict([input_message])

# Displaying result
print("The input message is classified as:", prediction[0])
