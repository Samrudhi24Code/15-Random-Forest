# README: Random Forest Model for Movie Classification

# Overview

This project aims to predict whether a movie will win the 'Start_Tech_Oscar' award based on specific features such as 3D availability and genre. A Random Forest Classifier is used to build the model for this binary classification task.

# Objective

The primary objective is to:

Maximize the accuracy of predictions for unseen data.

Minimize errors by utilizing an ensemble learning method.

# Workflow

# 1. Data Loading

The dataset, movies_classification.csv, is loaded into a Pandas DataFrame using the pd.read_csv() function. The file path is specified as:

E:\Data Science\15-Random Forest\movies_classification.csv

# 2. Data Preprocessing

Dummy Variable Creation: Categorical columns (3D_available and Genre) are converted to dummy variables using one-hot encoding to make them suitable for machine learning algorithms.

Avoiding Dummy Variable Trap: drop_first=True is used to omit the first category for each categorical variable.

# 3. Splitting the Dataset

The data is split into training and testing sets using an 80-20 ratio to:

Train the model on the majority of the data.

Test the model’s performance on unseen data.

# 4. Model Selection and Training

A Random Forest Classifier is used for this task. Key hyperparameters include:

n_estimators=500: Builds 500 decision trees for better accuracy.

n_jobs=1: Utilizes one CPU core for computation.

random_state=42: Ensures consistent results for reproducibility.

The model is trained using the fit() method on the training dataset.

# 5. Predictions

Training Dataset: Predictions are generated for the training data to evaluate training accuracy.

Test Dataset: Predictions are generated for the test data to evaluate generalization accuracy.

# 6. Model Evaluation

Performance metrics are calculated using:

Accuracy Score: Proportion of correctly predicted samples.

Confusion Matrix: Provides detailed insights into true positives, false positives, true negatives, and false negatives.

Results

# Training Dataset:

Accuracy Score: Evaluates how well the model has learned the training data.

Confusion Matrix: Highlights any overfitting issues if training accuracy is too high.

# Test Dataset:

Accuracy Score: Determines the model’s ability to generalize to unseen data.

Confusion Matrix: Identifies the distribution of correct and incorrect predictions.

Code Structure

Libraries Used

pandas: For data loading and preprocessing.

sklearn.model_selection: For splitting the dataset into training and testing sets.

sklearn.ensemble: For building the Random Forest Classifier.

sklearn.metrics: For evaluating the model.

Key Functions

pd.get_dummies():

Converts categorical variables into dummy variables.

Ensures the dataset is ready for numerical computations.

train_test_split():

Splits the data into training and testing sets.

Test size is set to 20%.

RandomForestClassifier:

Builds an ensemble model of decision trees.

Handles non-linear relationships in the data effectively.

accuracy_score():

Measures the proportion of correct predictions.

confusion_matrix():

Provides detailed performance metrics for classification.

File Structure

movies_classification.csv: Input dataset.

random_forest_classification.py: Python script containing the code.

# Usage

Install the necessary Python libraries:

pip install pandas scikit-learn

Update the file path for movies_classification.csv in the code.

Run the Python script to:

Train the Random Forest model.

Generate predictions.

Evaluate the model's performance.

# Recommendations

Hyperparameter Tuning:
Experiment with different values of n_estimators, max_depth, and min_samples_split for better performance.

Feature Engineering:
Add or transform features to enhance the model's predictive power.

Cross-Validation:
Use k-fold cross-validation to validate the model's performance across different data splits.

# Conclusion

The Random Forest Classifier is a robust model that effectively handles categorical and numerical data. It provides high accuracy and detailed performance metrics, making it suitable for predicting movie awards. Further optimization and feature engineering can enhance the model's performance.


