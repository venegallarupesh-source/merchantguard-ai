\# 🛡️ MerchantGuard AI

🔗 Live Demo: https://merchantguard-ai-dzdzbaghveokmqdi5ytwop.streamlit.app/



\*\*Explainable AI-powered merchant risk scoring system\*\* — built for Razorpay's AI Buildathon 2026 (AI Risk Manager track).



\## The Problem

Payment platforms like Razorpay onboard thousands of merchants. Some pose risk — high refund rates, sudden volume spikes, chargeback patterns — that can lead to fraud, defaults, or financial loss. Manually reviewing every merchant doesn't scale.



\## The Solution

MerchantGuard AI scores every merchant's risk (0-100) using transaction behavior, and explains why each score was given, not just what it is.

🔗 \*\*Live Demo:\*\* https://merchantguard-ai-dzdzbaghveokmqdi5ytwop.streamlit.app/



!\[Architecture](architecture\_diagram.png)



\## How It Works

\## How It Works

1\. Data - Merchant transaction data (volume, refund rate, chargeback rate, account age, category, growth patterns)

2\. Model - XGBoost classifier trained to predict risk (ROC-AUC: 0.95)

3\. Explainability - SHAP values break down exactly which factors drove each merchant's score

4\. Dashboard - Interactive Streamlit app to browse, filter, and explain any merchant's risk in real time



\## Tech Stack

Python, XGBoost, SHAP, Streamlit, Pandas, Scikit-learn

\## Results

\- \*\*0.95 ROC-AUC\*\* on held-out test data

\- \*\*82% recall\*\* on risky merchants

\- Fully explainable - every prediction traceable to specific behavioral signals

\- \*\*False-positive cost:\*\* Every wrongly-flagged safe merchant means delayed payouts and manual review overhead — the model is tuned to balance catching real risk against this business cost, not maximize accuracy alone.

\- \*\*Cost-optimized threshold:\*\* Instead of a default 0.5 cutoff, the decision threshold was optimized against estimated business cost (₹50,000 per missed risky merchant vs ₹2,000 per false alarm) — reducing total estimated cost by \*\*62.8%\*\* (₹10.98L → ₹4.08L on the test set).



!\[Threshold Cost Curve](threshold\_cost\_curve.png)

\## Run It Locally

pip install pandas numpy scikit-learn xgboost shap streamlit matplotlib

streamlit run app.py

\## Edge Case Testing

The model was stress-tested against deliberately unusual merchant profiles:

\- Correctly flagged a brand-new account with a suspicious volume spike (0.82 risk)

\- Correctly flagged a long-trusted merchant with one genuinely bad month (0.95 risk)

\- Correctly passed a "high-risk category" (Crypto/Trading) merchant with clean behavior — confirming the model judges actual behavior, not category stereotypes

\- \*\*Known limitation:\*\* Dormant, zero-activity merchants score as low-risk. In production, this would need a separate "sleeper account" monitoring layer, since fraud rings sometimes hold accounts inactive before use.

\## Scope

MerchantGuard AI is strictly a \*\*defensive risk-scoring tool\*\*. It flags merchant behavior for human review — it does not take autonomous action, block transactions, or make final decisions. All outputs are advisory signals for a risk analyst.



\## Author

Built solo by Rupesh for Razorpay AI Buildathon 2026.

