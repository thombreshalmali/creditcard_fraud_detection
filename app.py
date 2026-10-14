
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Dashboard title
st.title("💳 Credit Card Fraud Detection System")

st.write(
    "Machine Learning based system to identify "
    "potentially fraudulent credit card transactions."
)

st.info("Educational prototype — not for real banking decisions.")

# Load dataset and trained model
@st.cache_data
def load_data():
    return pd.read_csv("data/creditcard.csv")



@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

df = load_data()

# Model selection
st.sidebar.header("⚙️ Model Settings")

model_choice = st.sidebar.selectbox(
    "Select Machine Learning Model",
    ["Logistic Regression", "Random Forest"]
)

if model_choice == "Logistic Regression":
    model_path = "models/fraud_model.pkl"
else:
    model_path = "models/random_forest_model.pkl"

model = load_model(model_path)

st.sidebar.success(f"Active Model: {model_choice}")

st.sidebar.subheader("Fraud Detection Settings")

fraud_threshold = st.sidebar.slider(
    "Fraud Detection Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05
)

st.sidebar.caption(
    "Lower thresholds flag more transactions. "
    "Higher thresholds flag fewer."
)


st.sidebar.write(
    f"Current Threshold: {fraud_threshold:.2f}"
)


# Dashboard metrics
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Transactions", len(df))

with col2:
    st.metric("Legitimate Transactions", (df["Class"] == 0).sum())

with col3:
    st.metric("Fraud Transactions", (df["Class"] == 1).sum())

st.divider()

# Transaction selection
st.subheader("Transaction Fraud Prediction")

transaction_id = st.number_input(
    "Enter Transaction Row Number",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)

transaction = df.iloc[[transaction_id]]

# Display transaction information
st.write("Selected Transaction Details:")
st.dataframe(transaction)

# Predict button
if st.button("Predict Fraud"):
    
    features = transaction.drop(columns=["Class"])

    probability = float(
        model.predict_proba(features)[0][1]
    )

    prediction = int(probability >= fraud_threshold)

    st.write(f"Fraud Score: {probability:.4f}")
    st.write(f"Selected Threshold: {fraud_threshold:.2f}")
    st.write(f"Predicted Class: {prediction}")

    if prediction == 1:
        st.error("⚠️ Potential Fraudulent Transaction")
    else:
        st.success("✅ Legitimate Transaction")


# Additional Dashboard Visualizations

st.divider()
st.subheader("📊 Fraud Detection Analytics")

# Transaction class distribution
st.write("### Transaction Distribution")

class_counts = df["Class"].value_counts().sort_index()

distribution = pd.DataFrame({
    "Transaction Type": ["Legitimate", "Fraud"],
    "Count": [
        int(class_counts.get(0, 0)),
        int(class_counts.get(1, 0))
    ]
})

st.bar_chart(
    distribution.set_index("Transaction Type")
)

# Fraud percentage
fraud_percentage = (
    (df["Class"] == 1).sum() / len(df)
) * 100

st.metric(
    "Fraud Percentage in Dataset",
    f"{fraud_percentage:.3f}%"
)

st.divider()

# Transaction probability analysis
st.subheader("🔍 Transaction Risk Analysis")

features = transaction.drop(columns=["Class"])

fraud_probability = float(
    model.predict_proba(features)[0][1]
)

st.write("Model Estimated Fraud Probability")

st.progress(fraud_probability)

st.write(f"Fraud Probability: {fraud_probability:.2%}")

st.caption(
    "This is a model score, not a guaranteed or "
    "calibrated probability of fraud."
)

# Prediction comparison
st.subheader("Prediction vs Actual Result")

predicted_class = int(fraud_probability >= fraud_threshold)
actual_class = int(transaction["Class"].iloc[0])

comparison = pd.DataFrame({
    "Type": ["Predicted", "Actual"],
    "Result": [
        "Fraud" if predicted_class == 1 else "Legitimate",
        "Fraud" if actual_class == 1 else "Legitimate"
    ]
})

st.table(comparison)

if predicted_class == actual_class:
    st.success("Prediction matches the dataset label.")
else:
    st.warning("Prediction differs from the dataset label.")

st.divider()

st.subheader("📊 Machine Learning Model Comparison")

comparison_path = Path("results/model_comparison.csv")

if comparison_path.exists():
    comparison_df = pd.read_csv(comparison_path)

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    st.write("### F1-Score Comparison")

    chart_data = comparison_df.set_index("Model")[["F1 Score"]]

    st.bar_chart(chart_data)

else:
    st.warning(
        "Model comparison results not found. "
        "Run compare_models.py first."
    )

# Step 16 - Bulk Transaction Fraud Detection

st.divider()
st.subheader("📁 Bulk Fraud Detection Using CSV")

st.write(
    "Upload a CSV file containing credit card "
    "transactions to detect potential fraud."
)

uploaded_file = st.file_uploader(
    "Upload Transaction CSV File",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)

        st.write("Uploaded Dataset Preview")
        st.dataframe(uploaded_df.head())

        # Get the feature names expected by the model
        expected_features = list(model.feature_names_in_)

        # Check for missing columns
        missing_columns = [
            col for col in expected_features
            if col not in uploaded_df.columns
        ]

        if missing_columns:
            st.error(
                f"Missing required columns: {missing_columns}"
            )
        else:
            # Select only required features in correct order
            X_uploaded = uploaded_df[expected_features].copy()

            # Validate numeric and missing values
            if X_uploaded.isnull().any().any():
                st.error(
                    "Uploaded data contains missing values."
                )
            elif not all(
                pd.api.types.is_numeric_dtype(X_uploaded[col])
                for col in expected_features
            ):
                st.error(
                    "All transaction features must be numeric."
                )
            else:
                if st.button("Analyze Uploaded Transactions"):

                    probabilities = model.predict_proba(X_uploaded )[:, 1]
                    predictions = (
                        probabilities >= fraud_threshold
                    ).astype(int)

                    results_df = uploaded_df.copy()

                    results_df["Predicted_Class"] = predictions
                    results_df["Fraud_Score"] = probabilities

                    # Display results
                    st.success("Analysis completed!")

                    total = len(results_df)
                    flagged = int(
                        (results_df["Predicted_Class"] == 1).sum()
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Transactions Analyzed", total)

                    with col2:
                        st.metric("Flagged Transactions", flagged)

                    st.dataframe(results_df)

                    # Download results
                    csv_data = results_df.to_csv(index=False)

                    st.download_button(
                        label="⬇️ Download Prediction Results",
                        data=csv_data,
                        file_name="fraud_predictions.csv",
                        mime="text/csv"
                    )

    except Exception as e:
        st.error(f"Unable to process CSV: {e}")
