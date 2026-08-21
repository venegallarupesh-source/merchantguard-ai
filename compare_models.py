import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

df = pd.read_csv('merchant_data.csv')
le = LabelEncoder()
df['business_category_encoded'] = le.fit_transform(df['business_category'])

feature_cols = [
    'business_category_encoded', 'account_age_days', 'monthly_txn_count',
    'avg_transaction_value', 'monthly_txn_volume', 'refund_rate',
    'chargeback_rate', 'volume_growth_rate_30d', 'customer_complaint_count'
]
X = df[feature_cols]
y = df['is_risky']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

# Model 1: XGBoost (our current model)
xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_proba)
xgb_cv = cross_val_score(xgb_model, X, y, cv=5, scoring='roc_auc')

print(f"\nXGBoost:")
print(f"  Test ROC-AUC: {xgb_auc:.4f}")
print(f"  5-Fold CV ROC-AUC: {xgb_cv.mean():.4f} (+/- {xgb_cv.std():.4f})")

# Model 2: Random Forest
rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf_model.fit(X_train, y_train)
rf_proba = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_proba)
rf_cv = cross_val_score(rf_model, X, y, cv=5, scoring='roc_auc')

print(f"\nRandom Forest:")
print(f"  Test ROC-AUC: {rf_auc:.4f}")
print(f"  5-Fold CV ROC-AUC: {rf_cv.mean():.4f} (+/- {rf_cv.std():.4f})")

print("\n" + "=" * 60)
if xgb_auc >= rf_auc:
    print(f"WINNER: XGBoost (higher test ROC-AUC: {xgb_auc:.4f} vs {rf_auc:.4f})")
else:
    print(f"WINNER: Random Forest (higher test ROC-AUC: {rf_auc:.4f} vs {xgb_auc:.4f})")
print("=" * 60)
print("\nNote: XGBoost is also preferred here because it integrates")
print("natively with SHAP TreeExplainer for fast, exact explanations.")