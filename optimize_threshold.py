import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve

model = joblib.load('risk_model.pkl')
le = joblib.load('label_encoder.pkl')
feature_cols = joblib.load('feature_cols.pkl')

df = pd.read_csv('merchant_data.csv')
df['business_category_encoded'] = le.transform(df['business_category'])
X = df[feature_cols]
y = df['is_risky']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
y_proba = model.predict_proba(X_test)[:, 1]

# Business cost assumptions (editable - these are realistic estimates)
COST_FALSE_NEGATIVE = 50000  # missing a risky merchant -> fraud/default loss
COST_FALSE_POSITIVE = 2000   # wrongly flagging a safe merchant -> lost revenue, review overhead

thresholds = np.arange(0.05, 0.95, 0.01)
costs = []

for t in thresholds:
    y_pred = (y_proba >= t).astype(int)
    fn = np.sum((y_pred == 0) & (y_test == 1))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    total_cost = fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE
    costs.append(total_cost)

costs = np.array(costs)
optimal_idx = np.argmin(costs)
optimal_threshold = thresholds[optimal_idx]
optimal_cost = costs[optimal_idx]

# Compare to default 0.5 threshold
default_idx = np.argmin(np.abs(thresholds - 0.5))
default_cost = costs[default_idx]

print("=" * 60)
print("THRESHOLD COST OPTIMIZATION")
print("=" * 60)
print(f"Cost assumptions: False Negative = Rs.{COST_FALSE_NEGATIVE:,} | False Positive = Rs.{COST_FALSE_POSITIVE:,}")
print(f"\nDefault threshold (0.50): Total cost = Rs.{default_cost:,.0f}")
print(f"Optimal threshold ({optimal_threshold:.2f}): Total cost = Rs.{optimal_cost:,.0f}")
print(f"Savings by optimizing: Rs.{default_cost - optimal_cost:,.0f} ({(1 - optimal_cost/default_cost)*100:.1f}% reduction)")

plt.figure(figsize=(8, 5))
plt.plot(thresholds, costs, color='#D85A30', linewidth=2)
plt.axvline(optimal_threshold, color='#1D9E75', linestyle='--', label=f'Optimal threshold = {optimal_threshold:.2f}')
plt.axvline(0.5, color='gray', linestyle=':', label='Default threshold = 0.50')
plt.xlabel('Classification Threshold')
plt.ylabel('Total Business Cost (Rs.)')
plt.title('Cost-Optimized Risk Threshold Selection')
plt.legend()
plt.tight_layout()
plt.savefig('threshold_cost_curve.png', dpi=150, bbox_inches='tight')
print("\nSaved threshold_cost_curve.png")

joblib.dump(optimal_threshold, 'optimal_threshold.pkl')