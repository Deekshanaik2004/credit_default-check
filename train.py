# train_model.py

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

# Load your dataset
data = pd.read_csv("credit_data_500.csv")

# Split data
X = data.drop("Defaulted", axis=1)
y = data["Defaulted"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "xgb_model.pkl")

print("Model trained and saved as xgb_model.pkl")
