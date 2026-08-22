import pandas as pd


def flag_dormant_accounts(df):
    df = df.copy()

    low_activity_threshold = df['monthly_txn_count'].quantile(0.05)
    sharp_decline_threshold = df['volume_growth_rate_30d'].quantile(0.05)

    is_established = df['account_age_days'] > 180
    is_low_activity = df['monthly_txn_count'] <= low_activity_threshold
    is_sharp_decline = df['volume_growth_rate_30d'] <= sharp_decline_threshold

    df['is_dormant_risk'] = is_established & is_low_activity & is_sharp_decline

    return df


if __name__ == "__main__":
    df = pd.read_csv('merchant_data.csv')
    df = flag_dormant_accounts(df)
    dormant_count = df['is_dormant_risk'].sum()
    print(f"Flagged {dormant_count} merchants as dormant risk")
    print(df[df['is_dormant_risk']][['merchant_id', 'account_age_days', 'monthly_txn_count', 'volume_growth_rate_30d']].head(10))