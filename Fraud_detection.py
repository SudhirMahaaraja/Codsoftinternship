import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the train and test datasets
train_df = pd.read_csv("fraudTrain.csv")
test_df = pd.read_csv("fraudTest.csv")

# Downsample the majority class to balance the dataset
no_fraud = train_df[train_df["is_fraud"] == 0]
fraud = train_df[train_df["is_fraud"] == 1]

no_fraud_downsampled = resample(no_fraud, replace=False, n_samples=len(fraud), random_state=42)
downsampled_df = pd.concat([no_fraud_downsampled, fraud])

# Prepare the features and target variable
X = downsampled_df.drop("is_fraud", axis=1)
y = downsampled_df["is_fraud"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Select only numeric columns for scaling
numeric_columns = X_train.select_dtypes(include=['float64', 'int64']).columns

# Standardize only numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numeric_columns])
X_test_scaled = scaler.transform(X_test[numeric_columns])

# Train the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# Make predictions on the test set
X_test_numeric = X_test[numeric_columns]  # Select numeric columns for prediction
y_pred = rf_classifier.predict(scaler.transform(X_test_numeric))

# Filter the test dataset to have the same number of rows as the predictions
test_df_filtered = test_df.iloc[:3003]

# Create the predictions DataFrame
predictions_df = pd.DataFrame({"trans_date_trans_time": test_df_filtered["trans_date_trans_time"],
                               "merchant": test_df_filtered["merchant"],
                               "category": test_df_filtered["category"],
                               "amt": test_df_filtered["amt"],
                               "first": test_df_filtered["first"],
                               "last": test_df_filtered["last"],
                               "job": test_df_filtered["job"],
                               "dob": test_df_filtered["dob"],
                               "trans_num": test_df_filtered["trans_num"],
                               "predicted_fraud": y_pred})


# Concatenate the original test data with the predictions DataFrame
output_df = pd.concat([test_df, predictions_df["predicted_fraud"]], axis=1)

# # Select the desired columns for the output CSV
# selected_columns = ["trans_date_trans_time", "merchant", "category", "amt", "first", "last", "job", "dob", "trans_num", "predicted_fraud"]
#
# # Save the selected columns to a new CSV file
# predictions_df[selected_columns].to_csv("fraud_predictions.csv", index=False)
# Filter for transactions with predicted fraud (predicted_fraud = 1)
filtered_df = predictions_df[predictions_df["predicted_fraud"] == 1]

# Select the desired columns for the output CSV
selected_columns = ["trans_date_trans_time", "merchant", "category", "amt", "first", "last", "job", "dob", "trans_num", "predicted_fraud"]

# Save the filtered data with selected columns to a new CSV file
filtered_df[selected_columns].to_csv("fraud_predictions.csv", index=False)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
