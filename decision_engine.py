def get_risk_level(score):
    """Convert a 0-100 risk score into a human-readable risk level."""
    if score <= 25:
        return "Low", "🟢"
    elif score <= 50:
        return "Medium", "🟡"
    elif score <= 75:
        return "High", "🟠"
    else:
        return "Critical", "🔴"

def get_recommended_action(score, chargeback_rate):
    """Decide the recommended next action based on risk score and business rules."""
    level, emoji = get_risk_level(score)

    # Business rule: extremely high chargeback rate escalates regardless of score
    if chargeback_rate > 0.10 and score > 40:
        return "Enhanced Investigation", "🔴", "Escalated due to elevated chargeback rate combined with risk score"

    if level == "Low":
        return "Approve", "🟢", "Low risk profile - no action needed"
    elif level == "Medium":
        return "Monitor", "🟡", "Some risk signals present - add to watchlist"
    elif level == "High":
        return "Manual Review", "🟠", "Multiple risk signals present - needs analyst review"
    else:
        return "Enhanced Investigation", "🔴", "Critical risk score - requires deep investigation"

if __name__ == "__main__":
    # Quick test
    test_cases = [
        (15, 0.02), (42, 0.03), (68, 0.05), (91, 0.02), (60, 0.15)
    ]
    print("Testing Decision Engine")
    print("=" * 60)
    for score, cb_rate in test_cases:
        action, emoji, reason = get_recommended_action(score, cb_rate)
        print(f"Score: {score} | Chargeback: {cb_rate*100:.0f}% -> {emoji} {action}")
        print(f"  Reason: {reason}\n")