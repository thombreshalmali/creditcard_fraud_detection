
import pandas as pd
import joblib

df = pd.read_csv("data/creditcard.csv")
model = joblib.load("models/random_forest_model.pkl")

X = df.drop(columns=["Class"])

probabilities = model.predict_proba(X)[:, 1]

candidates = pd.DataFrame({
    "Row": df.index,
    "Fraud_Score": probabilities,
    "Actual_Class": df["Class"]
})

candidates = candidates[
    (candidates["Fraud_Score"] > 0.05) &
    (candidates["Fraud_Score"] < 0.95)
]

print("\nTransactions suitable for threshold testing:")
print(candidates.head(10).to_string(index=False))
