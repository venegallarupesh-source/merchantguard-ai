import pandas as pd
import numpy as np
import joblib

model = joblib.load('risk_model.pkl')
le = joblib.load('label_encoder.pkl')
feature_cols = joblib.load('feature_cols.pkl')
optimal_threshold = joblib.load('optimal_threshold.pkl')

def score_merchant(merchant_dict):
    """Score a single hand-crafted merchant and return risk probability."""
    row = pd.DataFrame([merchant_dict])
    row['business_category_encoded'] = le.transform(row['business_category'])
    X = row[feature_cols]
    proba = model.predict_proba(X)[0, 1]
    flagged = proba >= optimal_threshold
    return proba, flagged

print("=" * 70)
print("EDGE CASE TESTING - Deliberately unusual merchants")
print("=" * 70)

edge_cases = {
    "Brand new account, huge volume (suspicious)": {
        'business_category': 'Retail', 'account_age_days': 5,
        'monthly_txn_count': 50, 'avg_transaction_value': 80000,
        'monthly_txn_volume': 4000000, 'refund_rate': 0.02,
        'chargeback_rate': 0.01, 'volume_growth_rate_30d': 300,
        'customer_complaint_count': 0
    },
    "Old trusted account, one bad month": {
        'business_category': 'Electronics', 'account_age_days': 1800,
        'monthly_txn_count': 500, 'avg_transaction_value': 3000,
        'monthly_txn_volume': 1500000, 'refund_rate': 0.25,
        'chargeback_rate': 0.08, 'volume_growth_rate_30d': 15,
        'customer_complaint_count': 5
    },
    "Zero activity, dormant merchant": {
        'business_category': 'Services', 'account_age_days': 900,
        'monthly_txn_count': 1, 'avg_transaction_value': 500,
        'monthly_txn_volume': 500, 'refund_rate': 0.0,
        'chargeback_rate': 0.0, 'volume_growth_rate_30d': -90,
        'customer_complaint_count': 0
    },
    "High-risk category but clean behavior": {
        'business_category': 'Crypto/Trading', 'account_age_days': 600,
        'monthly_txn_count': 300, 'avg_transaction_value': 5000,
        'monthly_txn_volume': 1500000, 'refund_rate': 0.01,
        'chargeback_rate': 0.005, 'volume_growth_rate_30d': 5,
        'customer_complaint_count': 0
    },
    "Perfect-looking merchant, tiny chargeback anomaly": {
        'business_category': 'Healthcare', 'account_age_days': 1200,
        'monthly_txn_count': 800, 'avg_transaction_value': 2000,
        'monthly_txn_volume': 1600000, 'refund_rate': 0.03,
        'chargeback_rate': 0.15, 'volume_growth_rate_30d': 2,
        'customer_complaint_count': 1
    },
}

for name, merchant in edge_cases.items():
    proba, flagged = score_merchant(merchant)
    status = "🔴 FLAGGED AS RISKY" if flagged else "🟢 PASSED AS SAFE"
    print(f"\n{name}")
    print(f"  Risk probability: {proba:.3f} | Threshold: {optimal_threshold:.2f}")
    print(f"  Result: {status}")

print("\n" + "=" * 70)
print("Review these results honestly - note any surprising or wrong calls")
print("=" * 70)