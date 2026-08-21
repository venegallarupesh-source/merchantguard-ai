import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="MerchantGuard AI", layout="wide")

# Load everything
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

# Sidebar filters
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

# Top metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Merchants", len(df))
col2.metric("Flagged as Risky", int(df['is_risky'].sum()))
col3.metric("Risk Rate", f"{df['is_risky'].mean()*100:.1f}%")

st.divider()

# Merchant table
st.subheader("Merchant Risk Overview")
display_cols = ['merchant_id', 'business_category', 'account_age_days',
                 'monthly_txn_volume', 'refund_rate', 'chargeback_rate', 'risk_score', 'is_risky']
st.dataframe(filtered_df[display_cols].sort_values('risk_score', ascending=False), use_container_width=True)

st.divider()

# Individual merchant explanation
st.subheader("🔍 Explain a Specific Merchant")
selected_merchant = st.selectbox("Select merchant ID", filtered_df['merchant_id'].tolist())

if selected_merchant:
    idx = df[df['merchant_id'] == selected_merchant].index[0]
    merchant_row = df.loc[idx]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Risk Score", f"{merchant_row['risk_score']:.1f}/100")
        st.write(f"**Category:** {merchant_row['business_category']}")
        st.write(f"**Account Age:** {merchant_row['account_age_days']} days")
        st.write(f"**Status:** {'🔴 Risky' if merchant_row['is_risky'] == 1 else '🟢 Safe'}")

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

st.divider()
st.caption("Built with XGBoost + SHAP | Razorpay AI Buildathon 2026")