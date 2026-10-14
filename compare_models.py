
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)

# Load dataset
df = pd.read_csv("data/creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

# Same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Define models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    
    # Save the trained Random Forest model
    if name == "Random Forest":
        import joblib
        from pathlib import Path

        Path("models").mkdir(exist_ok=True)

        joblib.dump(
            model,
            "models/random_forest_model.pkl"
        )

        print("Random Forest model saved successfully!")


    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "Average Precision": average_precision_score(
            y_test, probabilities
        ),
        "ROC-AUC": roc_auc_score(
            y_test, probabilities
        )
    })

# Display comparison
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON RESULTS")
print(results_df.to_string(index=False))

# Save comparison results
from pathlib import Path

Path("results").mkdir(exist_ok=True)

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nComparison saved successfully!")
