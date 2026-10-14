# Credit Card Fraud Detection System Using Machine Learning

## Project Overview
This project is a machine learning-based credit card fraud detection system developed using Python, Scikit-learn, and Streamlit.

The system analyzes historical credit card transactions and predicts whether a transaction is legitimate or potentially fraudulent.

**This is an educational prototype, not a production banking system.**

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Joblib
- Visual Studio Code

## Machine Learning Models
1. Logistic Regression
2. Random Forest Classifier

## Dataset
The project uses the Credit Card Fraud Detection dataset containing 284,807 transactions, including 492 fraudulent transactions.

## Features
- Individual transaction fraud prediction
- Fraud score visualization
- Model selection between Logistic Regression and Random Forest
- Adjustable fraud detection threshold
- Bulk CSV transaction analysis
- Downloadable prediction results
- Model evaluation and comparison charts

## Model Comparison

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Precision | 0.0610 | 0.9059 |
| Recall | 0.9184 | 0.7857 |
| F1-score | 0.1144 | 0.8415 |
| Average Precision | 0.7190 | 0.8629 |
| ROC-AUC | 0.9721 | 0.9573 |

These results were obtained using a stratified 80/20 train-test split. Model selection and threshold tuning should be validated separately before drawing final performance conclusions.

## How to Run

1. Create and activate a Python virtual environment.
2. Install the dependencies:

   `python -m pip install -r requirements.txt`

3. Place `creditcard.csv` inside the `data` folder.
4. Train the model if saved models are unavailable.
5. Start the dashboard:

   `python -m streamlit run app.py`

## Limitations
- The dataset is historical and highly imbalanced.
- Some transaction features are anonymized.
- The model may produce false alarms or miss fraudulent transactions.
- Fraud scores are not guaranteed to be calibrated probabilities.
- The system is intended for learning and demonstration only.

## Future Improvements
- Hyperparameter tuning
- Cross-validation and separate validation datasets
- Probability calibration
- Model explainability
- Improved dashboard design
- Deployment for demonstration purposes