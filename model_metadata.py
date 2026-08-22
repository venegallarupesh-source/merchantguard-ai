import json
import joblib
from datetime import datetime

feature_cols = joblib.load('feature_cols.pkl')
optimal_threshold = joblib.load('optimal_threshold.pkl')

metadata = {
    "model_name": "MerchantGuard Risk Model",
    "version": "1.0",
    "algorithm": "XGBoost Classifier",
    "training_date": datetime.now().strftime("%Y-%m-%d"),
    "dataset": "Synthetic prototype dataset (2000 merchants)",
    "feature_list": feature_cols,
    "selected_threshold": round(float(optimal_threshold), 4),
    "roc_auc_test": 0.9492,
    "roc_auc_cv_mean": 0.9583,
    "notes": "Threshold optimized against estimated business cost (FN=Rs.50000, FP=Rs.2000), not accuracy alone."
}

with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("Saved model_metadata.json")
print(json.dumps(metadata, indent=2))