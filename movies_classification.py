import pandas as pd

# Correct file path
df = pd.read_csv(r"E:\Data Science\15-Random Forest\movies_classification.csv")

# Business Objective: 
# To predict whether a movie will win the 'Start_Tech_Oscar' award based on certain features such as 3D availability and Genre.

# Display information about the DataFrame
df.info()

# Note: The dataset contains two categorical columns ("3D_available", "Genre"). 
# These need to be converted into numerical format using one-hot encoding, as machine learning algorithms require numeric inputs.

# Converting categorical variables into dummy variables (one-hot encoding) to prepare the data for modeling
df = pd.get_dummies(df, columns=["3D_available", "Genre"], drop_first=True) 
# Drop_first=True avoids the dummy variable trap by omitting the first category for each categorical variable.

# Assign input (predictors) and output (target) variables
# Predictors: All columns except 'Start_Tech_Oscar'
predictors = df.loc[:, df.columns != "Start_Tech_Oscar"]
# Target: 'Start_Tech_Oscar'
target = df["Start_Tech_Oscar"]

###################################################

# Splitting data into training and test sets
from sklearn.model_selection import train_test_split
# Minimizer: Minimize overfitting and maximize the generalization of the model.
# Solution: Use an 80-20 split for training and testing, where 80% is for training and 20% is for testing.
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.2, random_state=42)

###########################################

# Model selection: Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rand_for = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)

'''
Explanation of Parameters:
- n_estimators=500: Build 500 decision trees in the forest to maximize the model's accuracy.
- n_jobs=1: Utilize one CPU core for computation to avoid system overload (can be set to -1 for all available cores).
- random_state=42: Fixes the randomness for consistent results (important for reproducibility).

Objective of the Model:
- Minimize prediction error and maximize classification accuracy by learning patterns in the training data.
'''

# Train the Random Forest model using the training dataset
rand_for.fit(X_train, y_train)

# Generate predictions for the training dataset
pred_X_train = rand_for.predict(X_train)

# Generate predictions for the test dataset
pred_X_test = rand_for.predict(X_test)

#########################

# Evaluate model performance using accuracy score and confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix

# Test dataset evaluation
test_accuracy = accuracy_score(pred_X_test, y_test)  # Accuracy for test data
test_confusion = confusion_matrix(pred_X_test, y_test)  # Confusion matrix for test data

'''
Business Objective: 
- Maximize the model's ability to correctly predict 'Start_Tech_Oscar' awards on unseen data (test data).

Evaluation Metrics:
- Accuracy Score: Measures the proportion of correct predictions.
- Confusion Matrix: Provides detailed insights into true positives, false positives, true negatives, and false negatives.
'''

##########################################

# Training dataset evaluation
train_accuracy = accuracy_score(pred_X_train, y_train)  # Accuracy for training data
train_confusion = confusion_matrix(pred_X_train, y_train)  # Confusion matrix for training data

'''
Objective for Training Data:
- Minimize training error to ensure the model has correctly learned from the data.

Key Observations:
- If training accuracy is significantly higher than test accuracy, it may indicate overfitting.
'''

###############################
# Business Summary and Recommendations:
# Objective:
# - Build a model to predict the likelihood of winning the 'Start_Tech_Oscar' award for movies based on features such as Genre and 3D availability.
# - Maximize the accuracy while minimizing errors to ensure reliable predictions.

# Key Steps:
# 1. Data Preprocessing: Convert categorical features into numeric (one-hot encoding).
# 2. Train-Test Split: Use 80% for training and 20% for testing to balance between model training and validation.
# 3. Model Selection: Use Random Forest to leverage its ability to handle non-linear data and avoid overfitting due to ensemble methods.
# 4. Evaluation: Measure the accuracy and confusion matrix to assess the model's performance.
