# Predicting credit card approvals
Credit Card Approval Prediction using Machine Learning
📌 Project Overview

Commercial banks receive a large number of credit card applications daily. Manually evaluating these applications is time-consuming, error-prone, and inefficient. This project builds an automated credit card approval prediction system using machine learning techniques, similar to those used in real-world banking systems.

Using the Credit Card Approval dataset from the UCI Machine Learning Repository, we preprocess the data, explore it, and train a machine learning model to predict whether a credit card application will be approved (+) or denied (-).

📊 Dataset Description

Source: UCI Machine Learning Repository

Instances: 690 credit card applications

Features: 16 (anonymized for confidentiality)

Target Variable: Approval Status (+ for approved, - for denied)

Data Types:

Numerical (float & integer)

Categorical (object)

Challenges in Data:

Missing values marked as ?

Mixed data types

Features with different value ranges

🛠️ Project Workflow
1. Data Loading and Inspection

Loaded the dataset using pandas

Inspected structure, summary statistics, and data types

Identified missing values and mixed feature types

2. Train-Test Split & Feature Selection

Dropped less relevant features (DriversLicense, ZipCode)

Split data into:

Training set: 67%

Test set: 33%

Ensured no data leakage during preprocessing

3. Handling Missing Values

Replaced ? with NaN

Numerical columns: Imputed using mean values

Categorical columns: Imputed using the most frequent value

Verified that no missing values remained

4. Data Preprocessing

Converted categorical variables into numeric form using one-hot encoding

Aligned training and test datasets after encoding

Scaled features to a 0–1 range using MinMaxScaler

5. Model Building

Chosen model: Logistic Regression

Suitable due to correlated features

Commonly used for binary classification tasks

Model trained on the preprocessed training data

6. Model Evaluation

Evaluated using:

Accuracy score

Confusion matrix

Results:

Accuracy: 100%

Perfect classification of approved and denied applications on the test set

7. Hyperparameter Tuning

Used GridSearchCV with 5-fold cross-validation

Tuned parameters:

tol

max_iter

Best parameters found:

{'max_iter': 100, 'tol': 0.001}


Best cross-validation score: 1.0

Test set accuracy remained 100%

✅ Final Results

Successfully built a fully automated credit card approval predictor

Achieved perfect accuracy on the test dataset

Demonstrated essential data science techniques:

Data cleaning

Missing value imputation

Feature encoding

Feature scaling

Model training and evaluation

Hyperparameter optimization

📚 Technologies Used

Python

Pandas

NumPy

Scikit-learn

🚀 Conclusion

This project showcases a complete end-to-end machine learning classification pipeline, from raw data to a highly accurate predictive model. The approach mirrors real-world financial decision systems and highlights the importance of proper preprocessing and model tuning in achieving strong performance.
