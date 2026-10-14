from pathlib import Path

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

# Step 1: Load the dataset
data_path = Path("data/creditcard.csv")
df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Step 2: Separate features and target
X = df.drop(columns=["Class"])
y = df["Class"]

print("\nFraud transactions:", int(y.sum()))
print("Legitimate transactions:", int((y == 0).sum()))

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Step 4: Create the ML pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )),
])

# Step 5: Train the model
print("\nTraining the model...")
model.fit(X_train, y_train)

print("Training completed!")

# Step 6: Make predictions on unseen test data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Step 7: Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Legitimate", "Fraud"],
    zero_division=0,
))

print(
    "Average Precision:",
    round(average_precision_score(y_test, y_prob), 4),
)

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, y_prob), 4),
)

# Step 8: Save the trained model
model_path = Path("models/fraud_model.pkl")
model_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Saved location:", model_path)