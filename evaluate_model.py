
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)

# Load dataset
df = pd.read_csv("data/creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

# Recreate the same test split used during training
_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Load trained model
model = joblib.load("models/fraud_model.pkl")

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Display metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Average Precision:",
      average_precision_score(y_test, y_prob))

print("ROC-AUC:",
      roc_auc_score(y_test, y_prob))

# Create output folder
Path("results").mkdir(exist_ok=True)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"],
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Credit Card Fraud Detection - Confusion Matrix")

plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(
    y_test, y_prob
)

plt.figure(figsize=(7, 5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Fraud Detection - Precision Recall Curve")
plt.tight_layout()
plt.savefig("results/precision_recall_curve.png")
plt.show()

print("\nEvaluation charts saved in results/")
