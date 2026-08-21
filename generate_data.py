import pandas as pd
import numpy as np

np.random.seed(42)
n = 2000

categories = ['Retail', 'Electronics', 'Travel', 'Gaming', 'Food & Beverage',
              'Fashion', 'Healthcare', 'Education', 'Crypto/Trading', 'Services']

account_age_days = np.random.randint(10, 2000, n)
business_category = np.random.choice(categories, n)
monthly_txn_count = np.random.randint(20, 5000, n)
avg_transaction_value = np.random.gamma(2, 2000, n)
monthly_txn_volume = monthly_txn_count * avg_transaction_value
refund_rate = np.clip(np.random.beta(2, 20, n), 0, 1)
chargeback_rate = np.clip(np.random.beta(1.5, 40, n), 0, 1)
volume_growth_rate_30d = np.random.normal(5, 25, n)
customer_complaint_count = np.random.poisson(2, n)

risk_score_raw = (
    chargeback_rate * 400 +
    refund_rate * 150 +
    np.clip(volume_growth_rate_30d, 0, None) * 0.8 +
    customer_complaint_count * 3 +
    (2000 - np.clip(account_age_days, 0, 2000)) * 0.01 +
    np.random.normal(0, 8, n)
)

risk_score = np.clip(risk_score_raw, 0, 100)
is_risky = (risk_score > 65).astype(int)

df = pd.DataFrame({
    'merchant_id': [f'M{str(i).zfill(5)}' for i in range(n)],
    'business_category': business_category,
    'account_age_days': account_age_days,
    'monthly_txn_count': monthly_txn_count,
    'avg_transaction_value': avg_transaction_value.round(2),
    'monthly_txn_volume': monthly_txn_volume.round(2),
    'refund_rate': refund_rate.round(4),
    'chargeback_rate': chargeback_rate.round(4),
    'volume_growth_rate_30d': volume_growth_rate_30d.round(2),
    'customer_complaint_count': customer_complaint_count,
    'risk_score': risk_score.round(2),
    'is_risky': is_risky
})

df.to_csv('merchant_data.csv', index=False)
print(f"Generated {len(df)} merchants")
print(f"Risky merchants: {df['is_risky'].sum()} ({df['is_risky'].mean()*100:.1f}%)")
print(df.head())