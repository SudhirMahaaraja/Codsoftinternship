# Codsoftinternship
# Text Classification with Logistic Regression - Movie Genre Classification 

This repository contains Python code for performing text classification using logistic regression. The code demonstrates how to preprocess text data, train a logistic regression classifier, perform hyperparameter tuning using grid search, and make predictions on test data. Specifically, the classifier predicts the genre of a given text description.

## Installation

1. Clone the repository:

```
git clone https://github.com/your_username/your_repository.git
```

2. Install the required dependencies:

```
pip install pandas scikit-learn nltk
```

3. Download NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

### 1. Data Preparation

- Ensure that your training data is in the format of a CSV file with columns: 'ID', 'Title', 'Genre', 'Description'.
- Similarly, ensure that your test data is in the format of a CSV file with columns: 'ID', 'Title', 'Description'.
- Place your training and test data files in the root directory of the project.

### 2. Running the Code

Run the following command to execute the code:

```bash
python text_classification.py
```

### 3. Output

- The predictions made by the classifier on the test data will be saved in a file named `predicted_output_logistic_regression.txt`.
- A file named `genre_counts.txt` will be generated containing the count of all available genres in the training data.

Datasets
The datasets used in this project can be found on Kaggle at the following link: https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb

## Description

- `text_classification.py`: Python script containing the code for text classification. 
- `train_data.txt`: Training data file in CSV format.
- `test_data.txt`: Test data file in CSV format.



## Fraud Detection

This repository contains Python code for detecting fraud in financial transactions using a Random Forest classifier. The code demonstrates how to preprocess the data, downsample the majority class to balance the dataset, train a Random Forest classifier, make predictions, and evaluate the model's performance.

## Installation

1. Clone the repository:

```
git clone https://github.com/your_username/your_repository.git
```

2. Install the required dependencies:

```
pip install pandas scikit-learn
```

## Usage

### 1. Data Preparation

- Ensure that you have downloaded the `fraudTrain.csv` and `fraudTest.csv` files from the provided Kaggle dataset link.
- Place the dataset files in the root directory of the project.

### 2. Running the Code

Run the following command to execute the code:

```bash
python fraud_detection.py
```

### 3. Output

- The predictions made by the classifier on the test data will be saved in a file named `fraud_predictions.csv`.

# Datasets

The datasets used in this project can be found on Kaggle at the following link: [Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

## Description

- `fraud_detection.py`: Python script containing the code for fraud detection.
- `fraudTrain.csv`: Training data file.
- `fraudTest.csv`: Test data file.
