from pathlib import Path

import joblib
import pandas as pd

# Step 1: Define file paths
MODEL_PATH = Path("models/fraud_model.pkl")
DATA_PATH = Path("data/creditcard.csv")

# Step 2: Check that the files exist
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model not found. Run python train_model.py first."
    )

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset not found inside the data folder."
    )

# Step 3: Load the saved model and dataset
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# Step 4: Select one transaction
# We use a dataset row for testing.
# The actual Class label is excluded from model input.
transaction = df.drop(columns=["Class"]).iloc[[0]]

# Step 5: Predict the probability of fraud
fraud_probability = model.predict_proba(transaction)[0, 1]

# Step 6: Predict the class
prediction = model.predict(transaction)[0]

# Step 7: Display the result
print("\n--- Credit Card Fraud Detection ---")
print("Transaction index: 0")
print(f"Fraud probability: {fraud_probability:.2%}")

if prediction == 1:
    print("Prediction: POTENTIAL FRAUD")
else:
    print("Prediction: LIKELY LEGITIMATE")

# Display the true label only for checking the example
actual_class = int(df["Class"].iloc[0])

print("\nActual dataset label:", actual_class)
print("0 = Legitimate, 1 = Fraud")