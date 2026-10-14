import pandas as pd
from pathlib import Path

file_path = Path("data/creditcard.csv")

if not file_path.exists():
    print("Dataset not found!")
else:
    df = pd.read_csv(file_path)

    print("Project setup successful!")
    print("Dataset shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())
    print("\nFraud distribution:")
    print(df["Class"].value_counts())
    
    # Find fraudulent transaction row numbers
    fraud_rows = df.index[df["Class"] == 1].tolist()

    print("\nFirst 10 Fraud Transaction Row Numbers:")
    print(fraud_rows[:10])
