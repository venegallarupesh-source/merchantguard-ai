import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from drift_monitor import run_drift_report

df = pd.read_csv('merchant_data.csv')
le = joblib.load('label_encoder.pkl')
df['business_category_encoded'] = le.transform(df['business_category'])

feature_cols = joblib.load('feature_cols.pkl')
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['is_risky'])

monitor_features = ['chargeback_rate', 'refund_rate', 'monthly_txn_volume', 'volume_growth_rate_30d', 'account_age_days']

report = run_drift_report(train_df, test_df, monitor_features)
print(report)