def generate_investigation_report(merchant_id, score, level, action, impact_df, chargeback_rate, refund_rate, growth_rate):
    top3 = impact_df[impact_df['Impact on Risk'] > 0].head(3)

    findings = []
    for _, row in top3.iterrows():
        feat = row['Feature']
        val = row['Value']
        if feat == 'chargeback_rate':
            findings.append(f"Chargeback rate is {val*100:.1f}%, a key driver of elevated risk.")
        elif feat == 'refund_rate':
            findings.append(f"Refund rate is {val*100:.1f}%, higher than typical safe merchants.")
        elif feat == 'volume_growth_rate_30d':
            findings.append(f"Transaction volume grew {val:.1f}% in the last 30 days, a pattern associated with sudden risk.")
        elif feat == 'customer_complaint_count':
            findings.append(f"{int(val)} customer complaints logged, contributing to the risk assessment.")
        elif feat == 'account_age_days':
            findings.append(f"Account age of {int(val)} days is a contributing factor.")
        else:
            findings.append(f"{feat} (value: {val:.2f}) contributed to the risk score.")

    if not findings:
        findings.append("No single factor stood out strongly; risk is driven by a combination of smaller signals.")

    summary = f"Merchant {merchant_id} received a risk score of {score:.1f}/100 ({level}). "
    summary += "Multiple independent risk signals were identified. " if len(findings) > 1 else ""
    summary += f"Based on this evidence, the recommended action is: {action}."

    return findings, summary

if __name__ == "__main__":
    import pandas as pd
    fake_impact = pd.DataFrame({
        'Feature': ['chargeback_rate', 'refund_rate', 'volume_growth_rate_30d'],
        'Value': [0.15, 0.20, 65.0],
        'Impact on Risk': [3.5, 2.1, 1.8]
    })
    findings, summary = generate_investigation_report("M00016", 87.0, "Critical", "Enhanced Investigation", fake_impact, 0.15, 0.20, 65.0)
    print("FINDINGS:")
    for f in findings:
        print(f"  - {f}")
    print(f"\nSUMMARY:\n{summary}")