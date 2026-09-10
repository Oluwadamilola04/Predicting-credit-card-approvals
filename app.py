from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


st.set_page_config(
    page_title="Credit Card Approval Predictor",
    page_icon="💳",
    layout="centered",
)


@st.cache_resource
def train_model():
    data_path = Path(__file__).with_name("cc_approvals.data")
    applications = pd.read_csv(data_path, header=None)

    applications = applications.drop(columns=[11, 13])
    features = applications.drop(columns=[15]).replace("?", np.nan)
    features.columns = features.columns.astype(str)
    target = applications[15].map({"+": 1, "-": 0})

    numeric_columns = ["1", "2", "7", "10", "14"]
    categorical_columns = ["0", "3", "4", "5", "6", "8", "9", "12"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", MinMaxScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=100, tol=0.001)),
        ]
    )
    pipeline.fit(features, target)
    return pipeline, features, numeric_columns, categorical_columns


st.title("Credit Card Approval Predictor")
st.write(
    "Enter an example application to estimate whether it resembles an approved "
    "or denied application in the historical dataset."
)
st.warning(
    "This is an educational demonstration, not a real credit decision system. "
    "The dataset is small, anonymized, and not suitable for real lending decisions."
)

try:
    model, training_features, numeric_columns, categorical_columns = train_model()
except FileNotFoundError:
    st.error("The dataset file cc_approvals.data could not be found.")
    st.stop()

with st.form("application_form"):
    st.subheader("Application details")
    st.caption("The original dataset uses anonymized column names, so the fields are shown as features.")

    with st.expander("What do these features mean?", expanded=True):
        st.markdown(
            """
            The dataset hides the original names, but the columns are commonly understood as:

            | Feature | Likely meaning |
            | --- | --- |
            | 0 | Gender |
            | 1 | Age |
            | 2 | Debt or outstanding balance |
            | 3 | Marital status |
            | 4 | Existing bank customer status |
            | 5 | Education level |
            | 6 | Ethnicity |
            | 7 | Years employed |
            | 8 | Previous payment default |
            | 9 | Employment status |
            | 10 | Number of previous credit inquiries |
            | 12 | Citizenship status |
            | 14 | Annual income |

            The letters shown in some dropdowns are anonymized category codes, not values
            you need to interpret as real-world labels. The app uses them because those are
            the exact categories available in the historical dataset. Numeric fields use the
            same units and ranges found in that dataset.
            """
        )

    numeric_values = {}
    categorical_values = {}
    left_column, right_column = st.columns(2)

    for index, column in enumerate(numeric_columns):
        series = pd.to_numeric(training_features[column], errors="coerce")
        with (left_column if index % 2 == 0 else right_column):
            numeric_values[column] = st.number_input(
                f"Feature {column}",
                min_value=float(series.min()),
                max_value=float(series.max()),
                value=float(series.median()),
                key=f"numeric_{column}",
            )

    st.subheader("Additional information (optional)")
    st.caption(
        "You can leave these fields as 'Not provided'. The model will estimate the "
        "missing values using the most common values in the historical data."
    )
    left_column, right_column = st.columns(2)
    for index, column in enumerate(categorical_columns):
        options = ["Not provided"] + sorted(training_features[column].dropna().unique().tolist())
        with (left_column if index % 2 == 0 else right_column):
            categorical_values[column] = st.selectbox(
                f"Feature {column}", options, key=f"categorical_{column}"
            )

    submitted = st.form_submit_button("Predict approval", type="primary")

if submitted:
    application = {**numeric_values, **categorical_values}
    application_frame = pd.DataFrame([application]).replace("Not provided", np.nan)
    prediction = int(model.predict(application_frame)[0])
    probability = float(model.predict_proba(application_frame)[0, 1])

    if prediction == 1:
        st.success(f"Estimated result: Approved ({probability:.1%} approval probability)")
    else:
        st.error(f"Estimated result: Denied ({probability:.1%} approval probability)")

    st.caption("The probability is a model estimate, not a guarantee or a lending recommendation.")