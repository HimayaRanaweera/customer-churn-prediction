# Customer Churn Prediction using Machine Learning

This beginner-friendly machine learning project predicts whether a customer will churn (leave the service) based on their usage data.

## 📌 Project Features

- Exploratory Data Analysis (EDA)
- Data cleaning and label encoding
- Handling class imbalance using SMOTE
- Model training:
  - Decision Tree
  - Random Forest
  - XGBoost
- Accuracy, precision, recall, F1-score evaluation
- Model saved using Pickle

## 📁 Files Included

- `Customer_churn_prediction_using_ML.ipynb` - Main notebook
- `sample_data.csv` - Sample dataset (30 rows)
- `model.pkl` - Trained Random Forest model

## 🚀 How to Use

1. Clone this repo or download the notebook.
2. Open the `.ipynb` file in [Google Colab](https://colab.research.google.com/)
3. Run all cells to retrain the model, or load the saved one.
4. Use `model.predict()` on new customer data.

## 🧪 Sample Prediction Code

```python
import pickle
model = pickle.load(open(\"model.pkl\", \"rb\"))
sample_input = [[1, 0, 1, 0, 5, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 2, 1, 35.0, 180.5]]
prediction = model.predict(sample_input)
print(\"Will the customer churn?\", \"Yes\" if prediction[0]==1 else \"No\")
