import pandas as pd
import numpy as np


def calculate_psi(expected, actual, buckets=10):
    """
    Population Stability Index between two distributions.
    PSI < 0.10  -> Low drift
    0.10-0.25   -> Moderate drift
    PSI > 0.25  -> Significant drift
    """
    expected = np.array(expected)
    actual = np.array(actual)

    breakpoints = np.linspace(0, 100, buckets + 1)
    breakpoints = np.percentile(expected, breakpoints)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_pct = np.where(expected_counts == 0, 0.0001, expected_counts / len(expected))
    actual_pct = np.where(actual_counts == 0, 0.0001, actual_counts / len(actual))

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi


def psi_status(psi_value):
    if psi_value < 0.10:
        return "Low Drift", "🟢"
    elif psi_value < 0.25:
        return "Moderate Drift", "🟡"
    else:
        return "Significant Drift", "🔴"


def run_drift_report(train_df, current_df, features):
    results = []
    for feat in features:
        psi = calculate_psi(train_df[feat], current_df[feat])
        status, emoji = psi_status(psi)
        results.append({
            "Feature": feat,
            "PSI": round(psi, 4),
            "Status": f"{emoji} {status}"
        })
    return pd.DataFrame(results)