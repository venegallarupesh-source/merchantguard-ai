import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

# Load everything we saved earlier
model = joblib.load('risk_model.pkl')
le = joblib.load('label_encoder.pkl')
feature_cols = joblib.load('feature_cols.pkl')

# Load data again
df = pd.read_csv('merchant_data.csv')
df['business_category_encoded'] = le.transform(df['business_category'])
X = df[feature_cols]

# Create the SHAP explainer (TreeExplainer is fast and exact for XGBoost)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 1. Global summary plot - which features matter most overall
plt.figure()
shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved shap_summary.png - shows which features drive risk overall")

# 2. Explain one specific risky merchant in detail
risky_merchant_idx = df[df['is_risky'] == 1].index[0]
merchant_id = df.loc[risky_merchant_idx, 'merchant_id']

print(f"\n--- Explaining merchant {merchant_id} ---")
print(f"Risk score: {df.loc[risky_merchant_idx, 'risk_score']}")

merchant_shap = shap_values[risky_merchant_idx]
feature_impact = pd.DataFrame({
    'feature': feature_cols,
    'value': X.iloc[risky_merchant_idx].values,
    'shap_impact': merchant_shap
}).sort_values('shap_impact', key=abs, ascending=False)

print("\nTop factors driving this merchant's risk score:")
print(feature_impact.head(5).to_string(index=False))

print("\nDone! Check shap_summary.png to see it visually.")