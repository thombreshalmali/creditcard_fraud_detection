
import pandas as pd
from pathlib import Path

df = pd.read_csv("data/creditcard.csv")

# Select 10 legitimate and 10 fraudulent transactions
legitimate = df[df["Class"] == 0].sample(
    n=10, random_state=42
)

fraud = df[df["Class"] == 1].sample(
    n=10, random_state=42
)

sample = pd.concat([legitimate, fraud])

# Shuffle transactions
sample = sample.sample(frac=1, random_state=42)

Path("data").mkdir(exist_ok=True)

sample.to_csv(
    "data/sample_transactions.csv",
    index=False
)

print("Sample CSV created successfully!")
print("Total transactions:", len(sample))
