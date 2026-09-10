# Credit Card Approval Prediction

## Problem Statement
Banks receive many credit card applications. Reviewing every application manually takes time and can lead to inconsistent decisions. A bank needs a way to use information from previous applications to estimate whether a new application is likely to be approved or denied.

This project explores how machine learning can support that process by learning patterns from historical credit card applications. 

## Project Goal
Build a model that predicts whether a credit card application will be:

- **Approved (`+`)**
- **Denied (`-`)**

The project is intended as a machine learning demonstration. It is not a production lending system and should not be used to make real financial decisions.

## How the Project Solves the Problem
The model follows these steps:

1. Load historical application data.
2. Clean missing values and inconsistent data types.
3. Remove two features that are less useful for this prediction: `DriversLicense` and `ZipCode`.
4. Convert text-based categories into numbers so a model can use them.
5. Scale numeric values so features with larger numbers do not dominate the model.
6. Train a Logistic Regression classifier on the cleaned applications.
7. Evaluate its predictions on applications it has not seen during training.
8. Use cross-validation and grid search to find better model settings.

## Dataset
The project uses the [Credit Approval dataset from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/27/credit+approval).

- **Applications:** 690
- **Input features:** 15 after feature selection
- **Target:** approval status
- **Data:** a mixture of numerical and categorical values
- **Missing values:** represented by `?`
- **Feature names:** anonymized in the original dataset

The dataset is useful for demonstrating real data-cleaning challenges, but it is small and anonymized. Its results should therefore be interpreted cautiously.

## Data Preparation
The notebook handles the main data issues as follows:

- Replaces `?` with missing-value markers.
- Fills missing numerical values using training-set means.
- Fills missing categorical values using the most common training-set value for each column.
- Encodes categorical features with one-hot encoding.
- Scales features to a 0–1 range with `MinMaxScaler`.
- Separates the approval result before encoding to prevent target leakage.

Preprocessing statistics are learned from the training set and then applied to the test set. This prevents information from the test data from influencing the model during training.

## Model and Evaluation
The project uses **Logistic Regression**, a suitable baseline for a binary decision such as approved versus denied. The evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- 5-fold cross-validation

### Results

- **Baseline test accuracy:** 85.5%
- **Best 5-fold cross-validation score:** 86.8%
- **Macro F1-score:** approximately 0.86
- **Best parameters:** `max_iter=100`, `tol=0.001`

The confusion matrix and class-level metrics help show how well the model recognizes both approved and denied applications, rather than hiding performance behind one accuracy number.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Jupyter Notebook

## Project Files

- [`app.py`](app.py): Streamlit interface for testing example applications
- [`notebook.ipynb`](notebook.ipynb): data preparation, modeling, and evaluation
- [`cc_approvals.data`](cc_approvals.data): input dataset
- [`requirements.txt`](requirements.txt): Python environment packages

## Try the Demo

### Live App

**[Open the Credit Card Approval Predictor](https://predicting-credit-card-approvals-4hwsipbajtgjads4tngqkq.streamlit.app/)**

### Run Locally

From this project folder, install the dependencies and start Streamlit:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens a form where a user can enter an example application and receive an estimated approval or denial result. The prediction is intended for demonstration only.


## Key Takeaways

This project demonstrates a complete beginner-to-intermediate machine learning workflow:

- Translating a business problem into a classification task
- Preparing messy real-world data
- Preventing target leakage
- Training and tuning a model
- Evaluating performance with multiple metrics
- Communicating results and limitations clearly

The model achieves useful predictive performance on this dataset, but additional data, fairness analysis, explainability, monitoring, and regulatory review would be required before building a real credit approval system.
