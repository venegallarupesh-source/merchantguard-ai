import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from decision_engine import get_risk_level, get_recommended_action
from investigation_report import generate_investigation_report

st.set_page_config(page_title="MerchantGuard AI", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
    :root {
        --page-bg: #F6F8FC;
        --card-bg: #FFFFFF;
        --navy: #1E3A5F;
        --navy-hover: #16324F;
        --active-blue: #2563EB;
        --text-main: #334155;
        --text-secondary: #64748B;
        --border: #E2E8F0;
        --low-bg: #DCFCE7; --low-text: #15803D;
        --med-bg: #FEF3C7; --med-text: #B45309;
        --high-bg: #FFEDD5; --high-text: #C2410C;
        --crit-bg: #FEE2E2; --crit-text: #DC2626;
    }

    .stApp { background-color: var(--page-bg); }

    h1 { font-size: 2.2rem !important; color: var(--navy) !important; }
    h2 { font-size: 1.6rem !important; color: var(--navy) !important; }
    h3 { font-size: 1.2rem !important; color: var(--navy) !important; }
    p, span, label, .stMarkdown { color: var(--text-main); }
    .stCaption, [data-testid="stCaptionContainer"] { color: var(--text-secondary) !important; }

    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
        padding: 10px 16px;
        color: var(--text-secondary);
    }
    .stTabs [aria-selected="true"] {
        color: var(--navy) !important;
        border-bottom: 3px solid var(--active-blue) !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: var(--navy);
    }
    div[data-testid="stMetric"] {
        background-color: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }

    .stAlert { border-radius: 10px; }
    hr { margin: 1.5rem 0; border-color: var(--border); }

    .stButton > button {
        border-radius: 8px;
        border: 1px solid var(--navy);
        color: var(--navy);
        font-weight: 600;
        background-color: white;
    }
    .stButton > button:hover {
        background-color: var(--navy);
        color: white;
        border-color: var(--navy);
    }

    div[data-testid="stFormSubmitButton"] button {
        background-color: var(--navy) !important;
        color: white !important;
        border: 1px solid var(--navy) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: var(--navy-hover) !important;
        border-color: var(--navy-hover) !important;
    }
    div[data-testid="stFormSubmitButton"] button p {
        color: white !important;
    }

    div[data-baseweb="input"], div[data-baseweb="select"] > div {
        background-color: #F8FAFC;
        border-color: #CBD5E1 !important;
        border-radius: 8px;
    }
    div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within {
        border-color: var(--active-blue) !important;
    }

    div[data-testid="stJson"] {
        background-color: #F8FAFC;
        border-left: 4px solid var(--active-blue);
        border-radius: 6px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = joblib.load('risk_model.pkl')
    le = joblib.load('label_encoder.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    return model, le, feature_cols

@st.cache_data
def load_data():
    return pd.read_csv('merchant_data.csv')

model, le, feature_cols = load_assets()
df = load_data()
df['business_category_encoded'] = le.transform(df['business_category'])
X = df[feature_cols]
explainer = shap.TreeExplainer(model)

if 'audit_log' not in st.session_state:
    st.session_state.audit_log = []

st.title("🛡️ MerchantGuard AI")
st.caption("Explainable merchant risk scoring and decision support | Razorpay AI Buildathon 2026")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🆕 Live Risk Check", "📊 Overview", "🔍 Explain Merchant", "📈 Business Impact", "ℹ️ Model Info"
])

with tab1:
    st.subheader("Enter a new merchant's details for instant risk assessment")

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
            'business_category': in_category, 'account_age_days': in_age,
            'monthly_txn_count': in_txn_count, 'avg_transaction_value': in_avg_value,
            'monthly_txn_volume': in_txn_count * in_avg_value, 'refund_rate': in_refund_rate,
            'chargeback_rate': in_chargeback_rate, 'volume_growth_rate_30d': in_growth,
            'customer_complaint_count': in_complaints
        }])
        new_merchant['business_category_encoded'] = le.transform(new_merchant['business_category'])
        new_X = new_merchant[feature_cols]

        proba = model.predict_proba(new_X)[0, 1]
        new_score = proba * 100
        level, level_emoji = get_risk_level(new_score)
        action, action_emoji, reason, is_border = get_recommended_action(new_score, in_chargeback_rate)

        st.divider()
        rc1, rc2 = st.columns([1, 2])
        with rc1:
            score_color = "🔴" if new_score > 75 else "🟠" if new_score > 50 else "🟡" if new_score > 25 else "🟢"
            st.markdown(f"## {score_color} Risk Score: {new_score:.1f}/100")
            st.write(f"**Risk Level:** {level_emoji} {level}")
            confidence = abs(proba - 0.5) * 2 * 100
            st.write(f"**Prediction Certainty:** {confidence:.0f}%")
            st.markdown("---")
            if is_border:
                st.warning("⚠️ Borderline case - lower prediction certainty")
            st.markdown(f"### {action_emoji} Recommended Action")
            st.markdown(f"## {action}")
            st.caption(reason)
        with rc2:
            new_shap = explainer.shap_values(new_X)
            impact_df_live = pd.DataFrame({
                'Feature': feature_cols, 'Value': new_X.iloc[0].values, 'Impact on Risk': new_shap[0]
            }).sort_values('Impact on Risk', key=abs, ascending=False)
            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['red' if x > 0 else 'green' for x in impact_df_live['Impact on Risk']]
            ax.barh(impact_df_live['Feature'], impact_df_live['Impact on Risk'], color=colors)
            ax.set_xlabel('Impact on Risk Score')
            ax.set_title('Why this merchant was scored this way')
            plt.tight_layout()
            st.pyplot(fig)

with tab2:
    st.subheader("Merchant Portfolio Overview")

    high_risk_count = int((df['risk_score'] > 75).sum())
    avg_score = df['risk_score'].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("##### 🏢 Total Merchants")
        st.markdown(f"# {len(df)}")
    with col2:
        st.markdown("##### 🚩 Flagged Risky")
        st.markdown(f"# {int(df['is_risky'].sum())}")
    with col3:
        st.markdown("##### 🔴 Critical Cases")
        st.markdown(f"# {high_risk_count}")
    with col4:
        st.markdown("##### 📊 Avg Risk Score")
        st.markdown(f"# {avg_score:.0f}")

    st.divider()

    risk_filter = st.selectbox("Show merchants", ["All", "Risky only", "Safe only"])
    category_filter = st.multiselect("Filter by business category", df['business_category'].unique())

    filtered_df = df.copy()
    if risk_filter == "Risky only":
        filtered_df = filtered_df[filtered_df['is_risky'] == 1]
    elif risk_filter == "Safe only":
        filtered_df = filtered_df[filtered_df['is_risky'] == 0]
    if category_filter:
        filtered_df = filtered_df[filtered_df['business_category'].isin(category_filter)]

    display_cols = ['merchant_id', 'business_category', 'account_age_days',
                     'monthly_txn_volume', 'refund_rate', 'chargeback_rate', 'risk_score', 'is_risky']
    st.dataframe(filtered_df[display_cols].sort_values('risk_score', ascending=False), height=400)

with tab3:
    st.subheader("Explain a Specific Merchant's Score")
    selected_merchant = st.selectbox("Select merchant ID", df['merchant_id'].tolist())

    if selected_merchant:
        idx = df[df['merchant_id'] == selected_merchant].index[0]
        merchant_row = df.loc[idx]
        risk_level, level_emoji = get_risk_level(merchant_row['risk_score'])
        action, action_emoji, reason, is_border = get_recommended_action(merchant_row['risk_score'], merchant_row['chargeback_rate'])

        col1, col2 = st.columns([1, 2])
        with col1:
            score_color = "🔴" if merchant_row['risk_score'] > 75 else "🟠" if merchant_row['risk_score'] > 50 else "🟡" if merchant_row['risk_score'] > 25 else "🟢"
            st.markdown(f"## {score_color} Risk Score: {merchant_row['risk_score']:.1f}/100")
            st.write(f"**Risk Level:** {level_emoji} {risk_level}")
            merchant_proba = merchant_row['risk_score'] / 100
            confidence2 = abs(merchant_proba - 0.5) * 2 * 100
            st.write(f"**Prediction Certainty:** {confidence2:.0f}%")
            st.write(f"**Category:** {merchant_row['business_category']}")
            st.write(f"**Account Age:** {merchant_row['account_age_days']} days")
            st.markdown("---")
            if is_border:
                st.warning("⚠️ Borderline case - lower prediction certainty")
            st.markdown(f"### {action_emoji} Recommended Action")
            st.markdown(f"## {action}")
            st.caption(reason)
        with col2:
            shap_vals = explainer.shap_values(X.iloc[[idx]])
            impact_df = pd.DataFrame({
                'Feature': feature_cols, 'Value': X.iloc[idx].values, 'Impact on Risk': shap_vals[0]
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

            st.markdown("#### Top Risk Factors")
            feature_explanations = {
                'chargeback_rate': 'Higher chargeback rates strongly signal fraud or dissatisfaction.',
                'refund_rate': 'Elevated refunds indicate order or quality issues.',
                'volume_growth_rate_30d': 'Sudden volume spikes are a classic risk pattern.',
                'customer_complaint_count': 'More complaints directly raise risk concerns.',
                'account_age_days': 'Newer accounts carry less trust history.',
                'monthly_txn_volume': 'Unusual transaction volume can signal abnormal activity.',
                'monthly_txn_count': 'Transaction frequency patterns affect risk assessment.',
                'avg_transaction_value': 'Unusually high average ticket sizes add risk.',
                'business_category_encoded': 'Some categories carry inherently different baseline risk.'
            }
            top4 = impact_df.head(4).copy()
            top4['Reason'] = top4['Feature'].map(feature_explanations)
            top4['Effect'] = top4['Impact on Risk'].apply(lambda x: '🔴 Increases risk' if x > 0 else '🟢 Decreases risk')
            display_table = top4[['Feature', 'Effect', 'Reason']]
            st.table(display_table)

        st.divider()
        st.markdown("### 📋 AI Investigation Report")
        findings, summary = generate_investigation_report(
            selected_merchant, merchant_row['risk_score'], risk_level, action,
            impact_df, merchant_row['chargeback_rate'], merchant_row['refund_rate'],
            merchant_row['volume_growth_rate_30d']
        )
        st.markdown(f"**Merchant:** {selected_merchant} | **Risk Score:** {merchant_row['risk_score']:.1f}/100 | **Level:** {risk_level}")
        st.markdown("**Evidence Found:**")
        for f in findings:
            st.markdown(f"- {f}")
        st.info(f"**AI Recommendation:** {action}\n\n{summary}")

        st.divider()
        st.markdown("### 👤 Analyst Decision")
        st.caption("The AI recommends an action - a human analyst makes the final call. Every decision is logged to the audit trail.")

        st.write(f"**AI Recommendation:** {action}")
        st.write(f"**AI Risk Score:** {merchant_row['risk_score']:.0f}/100")

        decision_key = f"decision_{selected_merchant}"
        override_key = f"override_{selected_merchant}"

        dcol1, dcol2 = st.columns(2)
        with dcol1:
            if st.button("✅ Approve AI Recommendation", key=f"approve_{selected_merchant}"):
                record = {
                    "Merchant ID": selected_merchant,
                    "AI Risk Score": f"{merchant_row['risk_score']:.0f}/100",
                    "AI Recommendation": action,
                    "Final Analyst Decision": action,
                    "Reason": "AI recommendation approved as-is",
                    "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                }
                st.session_state.audit_log.append(record)
                st.session_state[decision_key] = action
                st.session_state[override_key] = False
                st.success("✅ DECISION RECORDED")
                st.json(record)
        with dcol2:
            if st.button("✏️ Override Decision", key=f"override_btn_{selected_merchant}"):
                st.session_state[override_key] = True

        if st.session_state.get(override_key, False):
            override_options = ["Approve", "Monitor", "Manual Review", "Enhanced Investigation"]
            new_decision = st.selectbox("Final Analyst Action", override_options, key=f"select_{selected_merchant}")
            override_reason = st.text_input("Reason for decision / override", key=f"reason_{selected_merchant}")
            if st.button("💾 Save Analyst Decision", key=f"save_{selected_merchant}"):
                record = {
                    "Merchant ID": selected_merchant,
                    "AI Risk Score": f"{merchant_row['risk_score']:.0f}/100",
                    "AI Recommendation": action,
                    "Final Analyst Decision": new_decision,
                    "Reason": override_reason if override_reason else "No reason provided",
                    "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
                }
                st.session_state.audit_log.append(record)
                st.session_state[decision_key] = new_decision
                st.session_state[f"final_reason_{selected_merchant}"] = override_reason
                st.session_state[override_key] = False
                st.success("✅ DECISION RECORDED")
                st.json(record)

        if decision_key in st.session_state and not st.session_state.get(override_key, False):
            final = st.session_state[decision_key]
            reason_saved = st.session_state.get(f"final_reason_{selected_merchant}", "")
            st.info(f"**Last Recorded Decision:** {final}" + (f" (Reason: {reason_saved})" if reason_saved else ""))

        if st.session_state.audit_log:
            st.divider()
            st.markdown("### 📋 Audit Trail")
            st.caption("Full history of analyst decisions across all reviewed merchants - every override is traceable with timestamp and reasoning.")
            st.dataframe(pd.DataFrame(st.session_state.audit_log), use_container_width=True)

        st.divider()
        st.markdown("### 📉 Risk Trend (Illustrative)")
        st.caption("Simulated 6-month trend based on this merchant's current risk profile - for demo purposes, not historical data.")

        np.random.seed(hash(selected_merchant) % 1000)
        current = merchant_row['risk_score']
        months = ['Month 1', 'Month 2', 'Month 3', 'Month 4', 'Month 5', 'Current']
        trend = np.linspace(max(current - 40, 0), current, 6) + np.random.normal(0, 5, 6)
        trend = np.clip(trend, 0, 100)
        trend[-1] = current

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(months, trend, marker='o', color='#D85A30', linewidth=2)
        ax2.axhline(65, color='gray', linestyle='--', alpha=0.5, label='Risk threshold zone')
        ax2.set_ylabel('Risk Score')
        ax2.set_title(f'{selected_merchant} - Risk Trend Over Time')
        ax2.legend()
        plt.tight_layout()
        st.pyplot(fig2)

with tab4:
    st.subheader("Threshold Trade-off & Business Cost")
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

    COST_FN = 50000
    COST_FP = 2000
    total_cost = fn * COST_FN + fp * COST_FP

    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric("Risky Caught", tp)
    bc2.metric("False Alarms", fp)
    bc3.metric("Risky Missed", fn)
    bc4.metric("Estimated Cost", f"Rs.{total_cost:,.0f}")

    st.markdown(f"At this threshold ({threshold_slider:.2f}): {tp} risky merchants caught, {fp} false alarms, {fn} missed. Estimated cost: Rs.{total_cost:,.0f}")

    if threshold_slider < 0.15:
        st.info("Lower threshold: catches more risk, more review workload.")
    elif threshold_slider > 0.5:
        st.warning("Higher threshold: less workload, higher risk of missed cases.")
    else:
        st.success("This range balances detection against workload reasonably well.")

with tab5:
    st.subheader("Model Information & Versioning")
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
    st.divider()
    st.markdown("**Scope:** MerchantGuard AI is a defensive, decision-support tool. It flags merchants for human review and does not take autonomous action.")

st.divider()
st.caption("Built with XGBoost + SHAP | Razorpay AI Buildathon 2026")