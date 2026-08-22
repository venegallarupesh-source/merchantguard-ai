import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from decision_engine import get_risk_level, get_recommended_action

st.set_page_config(page_title="MerchantGuard AI", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load('risk_model.pkl')
    le = joblib.load('label_encoder.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    return model, le, feature_cols

@st.cache_data
def load_data():
    df = pd.read_csv('merchant_data.csv')
    return df

model, le, feature_cols = load_assets()
df = load_data()
df['business_category_encoded'] = le.transform(df['business_category'])
X = df[feature_cols]

explainer = shap.TreeExplainer(model)

st.title("🛡️ MerchantGuard AI")
st.caption("Explainable merchant risk scoring for Razorpay AI Buildathon 2026")

st.divider()
st.subheader("🆕 Live Merchant Risk Check")
st.caption("Enter a new merchant's details to get an instant risk assessment")

with st.form("live_check_form"):
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        in_category = st.selectbox("Business Category", df['business_category'].unique())
        in_age = st.number_input("Account Age (days)", min_value=0, max_value=5000, value=365)
        in_txn_count = st.number_input("Monthly Transaction Count", min_value=1, max_value=10000, value=500)
    with fc2:
        in_avg_value = st.number_input("Avg Transaction Value (Rs.)", min_value=1.0, max_value=200000.0, value=2000.0)
        in_refund_rate = st.slider("Refund Rate", 0.0, 1.0, 0.05, 0.01)
        in_chargeback_rate = st.slider("Chargeback Rate", 0.0, 1.0, 0.02, 0.01)
    with fc3:
        in_growth = st.number_input("30-Day Volume Growth Rate (%)", min_value=-100.0, max_value=500.0, value=5.0)
        in_complaints = st.number_input("Customer Complaint Count", min_value=0, max_value=100, value=1)

    submitted = st.form_submit_button("🔍 ANALYZE MERCHANT")

if submitted:
    validation_errors = []
    if in_refund_rate > 1.0 or in_refund_rate < 0:
        validation_errors.append("Refund rate must be between 0 and 1.")
    if in_chargeback_rate > 1.0 or in_chargeback_rate < 0:
        validation_errors.append("Chargeback rate must be between 0 and 1.")
    if in_txn_count <= 0:
        validation_errors.append("Transaction count must be greater than 0.")
    if in_avg_value <= 0:
        validation_errors.append("Average transaction value must be greater than 0.")
    if in_age < 0:
        validation_errors.append("Account age cannot be negative.")

    if validation_errors:
        for err in validation_errors:
            st.error(f"⚠️ {err}")
        st.stop()

    new_merchant = pd.DataFrame([{
        'business_category': in_category,
        'account_age_days': in_age,
        'monthly_txn_count': in_txn_count,
        'avg_transaction_value': in_avg_value,
        'monthly_txn_volume': in_txn_count * in_avg_value,
        'refund_rate': in_refund_rate,
        'chargeback_rate': in_chargeback_rate,
        'volume_growth_rate_30d': in_growth,
        'customer_complaint_count': in_complaints
    }])
    new_merchant['business_category_encoded'] = le.transform(new_merchant['business_category'])
    new_X = new_merchant[feature_cols]

    proba = model.predict_proba(new_X)[0, 1]
    new_score = proba * 100
    level, level_emoji = get_risk_level(new_score)
    action, action_emoji, reason, is_border = get_recommended_action(new_score, in_chargeback_rate)

    rc1, rc2 = st.columns([1, 2])
    with rc1:
        st.metric("Risk Score", f"{new_score:.1f}/100")
        st.write(f"**Risk Level:** {level_emoji} {level}")
        st.markdown("---")
        if is_border:
            st.warning("⚠️ Borderline case - lower model confidence")
        st.markdown(f"### {action_emoji} Recommended Action")
        st.markdown(f"## {action}")
        st.caption(reason)

    with rc2:
        new_shap = explainer.shap_values(new_X)
        impact_df = pd.DataFrame({
            'Feature': feature_cols,
            'Value': new_X.iloc[0].values,
            'Impact on Risk': new_shap[0]
        }).sort_values('Impact on Risk', key=abs, ascending=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['red' if x > 0 else 'green' for x in impact_df['Impact on Risk']]
        ax.barh(impact_df['Feature'], impact_df['Impact on Risk'], color=colors)
        ax.set_xlabel('Impact on Risk Score')
        ax.set_title('Why this merchant was scored this way')
        plt.tight_layout()
        st.pyplot(fig)

st.divider()
st.subheader("Merchant Risk Overview")

st.sidebar.header("Filters")
risk_filter = st.sidebar.selectbox("Show merchants", ["All", "Risky only", "Safe only"])
category_filter = st.sidebar.multiselect("Business category", df['business_category'].unique())

filtered_df = df.copy()
if risk_filter == "Risky only":
    filtered_df = filtered_df[filtered_df['is_risky'] == 1]
elif risk_filter == "Safe only":
    filtered_df = filtered_df[filtered_df['is_risky'] == 0]
if category_filter:
    filtered_df = filtered_df[filtered_df['business_category'].isin(category_filter)]

col1, col2, col3 = st.columns(3)
col1.metric("Total Merchants", len(df))
col2.metric("Flagged as Risky", int(df['is_risky'].sum()))
col3.metric("Risk Rate", f"{df['is_risky'].mean()*100:.1f}%")

display_cols = ['merchant_id', 'business_category', 'account_age_days',
                 'monthly_txn_volume', 'refund_rate', 'chargeback_rate', 'risk_score', 'is_risky']
st.dataframe(filtered_df[display_cols].sort_values('risk_score', ascending=False), use_container_width=True)

st.divider()
st.subheader("🔍 Explain a Specific Merchant")
selected_merchant = st.selectbox("Select merchant ID", filtered_df['merchant_id'].tolist())

if selected_merchant:
    idx = df[df['merchant_id'] == selected_merchant].index[0]
    merchant_row = df.loc[idx]

    risk_level, level_emoji = get_risk_level(merchant_row['risk_score'])
    action, action_emoji, reason, is_border = get_recommended_action(merchant_row['risk_score'], merchant_row['chargeback_rate'])

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Risk Score", f"{merchant_row['risk_score']:.1f}/100")
        st.write(f"**Risk Level:** {level_emoji} {risk_level}")
        st.write(f"**Category:** {merchant_row['business_category']}")
        st.write(f"**Account Age:** {merchant_row['account_age_days']} days")

        st.markdown("---")
        if is_border:
            st.warning("⚠️ Borderline case - lower model confidence")
        st.markdown(f"### {action_emoji} Recommended Action")
        st.markdown(f"## {action}")
        st.caption(reason)

    with col2:
        shap_vals = explainer.shap_values(X.iloc[[idx]])
        impact_df = pd.DataFrame({
            'Feature': feature_cols,
            'Value': X.iloc[idx].values,
            'Impact on Risk': shap_vals[0]
        }).sort_values('Impact on Risk', key=abs, ascending=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['red' if x > 0 else 'green' for x in impact_df['Impact on Risk']]
        ax.barh(impact_df['Feature'], impact_df['Impact on Risk'], color=colors)
        ax.set_xlabel('Impact on Risk Score')
        ax.set_title(f'Why {selected_merchant} was scored this way')
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("#### In plain terms:")
        top_factor = impact_df.iloc[0]
        direction = "increased" if top_factor['Impact on Risk'] > 0 else "decreased"
        st.write(f"The biggest driver was **{top_factor['Feature']}** (value: {top_factor['Value']:.2f}), which **{direction}** this merchant's risk score the most.")

st.divider()
st.subheader("📈 Business Impact: Threshold Trade-off")
st.caption("Adjust the risk threshold to see the trade-off between catching risk and false alarms")

@st.cache_data
def get_test_predictions():
    df2 = pd.read_csv('merchant_data.csv')
    df2['business_category_encoded'] = le.transform(df2['business_category'])
    X2 = df2[feature_cols]
    y2 = df2['is_risky']
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)
    proba2 = model.predict_proba(X_test2)[:, 1]
    return y_test2.values, proba2

y_test_vals, proba_vals = get_test_predictions()

threshold_slider = st.slider("Risk Threshold", 0.05, 0.95, 0.13, 0.01)

pred_at_threshold = (proba_vals >= threshold_slider).astype(int)
tp = int(np.sum((pred_at_threshold == 1) & (y_test_vals == 1)))
fp = int(np.sum((pred_at_threshold == 1) & (y_test_vals == 0)))
fn = int(np.sum((pred_at_threshold == 0) & (y_test_vals == 1)))
tn = int(np.sum((pred_at_threshold == 0) & (y_test_vals == 0)))

COST_FN = 50000
COST_FP = 2000
total_cost = fn * COST_FN + fp * COST_FP

bc1, bc2, bc3, bc4 = st.columns(4)
bc1.metric("Risky Merchants Caught", tp)
bc2.metric("False Alarms", fp)
bc3.metric("Risky Merchants Missed", fn)
bc4.metric("Estimated Cost", f"₹{total_cost:,.0f}")

st.markdown(f"""
**At this threshold ({threshold_slider:.2f}):**
- {tp} genuinely risky merchants are correctly caught
- {fp} safe merchants are wrongly flagged (manual review overhead)
- {fn} risky merchants are missed (potential fraud/default loss)
- Estimated total business cost: **₹{total_cost:,.0f}** (assuming ₹50,000 per missed risk, ₹2,000 per false alarm)
""")

if threshold_slider < 0.15:
    st.info("Lower threshold: catches more risk, but generates more manual review workload.")
elif threshold_slider > 0.5:
    st.warning("Higher threshold: less workload, but risk of missing genuinely risky merchants increases.")
else:
    st.success("This range balances detection against review workload reasonably well.")

st.divider()

with st.expander("ℹ️ Model Information"):
    import json
    with open('model_metadata.json') as f:
        meta = json.load(f)
    mc1, mc2 = st.columns(2)
    with mc1:
        st.write(f"**Model:** {meta['model_name']} v{meta['version']}")
        st.write(f"**Algorithm:** {meta['algorithm']}")
        st.write(f"**Trained:** {meta['training_date']}")
        st.write(f"**Dataset:** {meta['dataset']}")
    with mc2:
        st.write(f"**Test ROC-AUC:** {meta['roc_auc_test']}")
        st.write(f"**5-Fold CV ROC-AUC:** {meta['roc_auc_cv_mean']}")
        st.write(f"**Decision Threshold:** {meta['selected_threshold']}")
    st.caption(meta['notes'])

st.caption("Built with XGBoost + SHAP | Razorpay AI Buildathon 2026")