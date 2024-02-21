import pandas as pd
import re
import nltk
import string
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

# Load the training data
train_path = "train_data.txt"
train_data = pd.read_csv(train_path, sep=' ::: ', names=['ID', 'Title', 'Genre', 'Description'], engine='python')

# Clean the text data
def clean_text(text):
    text = text.lower()  # Lowercase all characters
    text = re.sub(r'@\S+', '', text)  # Remove Twitter handles
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'pic.\S+', '', text)
    text = re.sub(r"[^a-zA-Z+']", ' ', text)  # Keep only characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text + ' ')  # Keep words with length > 1 only
    text = "".join([i for i in text if i not in string.punctuation])
    words = nltk.word_tokenize(text)
    stopwords = nltk.corpus.stopwords.words('english')  # Remove stopwords
    text = " ".join([i for i in words if i not in stopwords and len(i) > 2])
    text = re.sub("\s[\s]+", " ", text).strip()  # Remove repeated/leading/trailing spaces
    return text

train_data['Description'] = train_data['Description'].apply(clean_text)

# Separate the data by genre
drama_data = train_data[train_data['Genre'] == 'drama']
documentary_data = train_data[train_data['Genre'] == 'documentary']

# Oversample or duplicate drama and documentary data to have higher frequency
drama_data_oversampled = resample(drama_data, replace=True, n_samples=1000, random_state=42)
documentary_data_oversampled = resample(documentary_data, replace=True, n_samples=1000, random_state=42)

# Concatenate the oversampled data with the original data
train_data_oversampled = pd.concat([train_data, drama_data_oversampled, documentary_data_oversampled])

# Shuffle the data
train_data_oversampled = train_data_oversampled.sample(frac=1, random_state=42)

# Create TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer()
X_train = tfidf_vectorizer.fit_transform(train_data_oversampled['Description'])

# Split the data into features and target
y_train = train_data_oversampled['Genre']

# Initialize and train a Logistic Regression classifier
logistic_regression_classifier = LogisticRegression(max_iter=1000)

# Define hyperparameters for grid search
param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(logistic_regression_classifier, param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Initialize logistic regression classifier with best hyperparameters
best_logistic_regression_classifier = LogisticRegression(**best_params, max_iter=1000)

# Train the logistic regression classifier with best hyperparameters
best_logistic_regression_classifier.fit(X_train, y_train)

# Load the test data
test_path = "test_data.txt"
test_data = pd.read_csv(test_path, sep=' ::: ', names=['ID', 'Title', 'Description'], engine='python')

# Clean the test data
test_data['Description'] = test_data['Description'].apply(clean_text)

# Transform test data to TF-IDF vectors
X_test = tfidf_vectorizer.transform(test_data['Description'])

# Use the trained model to make predictions on the test data
X_test_predictions = best_logistic_regression_classifier.predict(X_test)
test_data['Predicted_Genre'] = X_test_predictions

# Output the predictions and descriptions to a text file
with open("predicted_output_logistic_regression.txt", "w") as output_file:
    for index, row in test_data.iterrows():
        output_file.write(f"{row['ID']} ::: {row['Title']} ::: {row['Predicted_Genre']} ::: {row['Description']}\n")

# Create a new text file for the count of all available genres
genre_counts = train_data['Genre'].value_counts()
genre_counts.to_csv("genre_counts.txt", header=False)
