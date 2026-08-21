import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import joblib

# Load data
df = pd.read_csv('merchant_data.csv')

# Encode business_category (text -> numbers, since models need numbers)
le = LabelEncoder()
df['business_category_encoded'] = le.fit_transform(df['business_category'])

# Define features (inputs) and target (what we predict)
feature_cols = [
    'business_category_encoded', 'account_age_days', 'monthly_txn_count',
    'avg_transaction_value', 'monthly_txn_volume', 'refund_rate',
    'chargeback_rate', 'volume_growth_rate_30d', 'customer_complaint_count'
]
X = df[feature_cols]
y = df['is_risky']

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the XGBoost model
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=['Not Risky', 'Risky']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model and encoder for later use
joblib.dump(model, 'risk_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(feature_cols, 'feature_cols.pkl')
print("\nModel saved as risk_model.pkl")