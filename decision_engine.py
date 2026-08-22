def get_risk_level(score):
    if score <= 25:
        return "Low", "🟢"
    elif score <= 50:
        return "Medium", "🟡"
    elif score <= 75:
        return "High", "🟠"
    else:
        return "Critical", "🔴"

def is_borderline(score, threshold_pct=5):
    """Check if score is close to a level boundary - signals lower confidence."""
    boundaries = [25, 50, 75]
    for b in boundaries:
        if abs(score - b) <= threshold_pct:
            return True
    return False

def get_recommended_action(score, chargeback_rate):
    level, emoji = get_risk_level(score)
    borderline = is_borderline(score)

    if chargeback_rate > 0.10 and score > 40:
        return "Enhanced Investigation", "🔴", "Escalated due to elevated chargeback rate combined with risk score", False

    if borderline:
        return "Manual Review", "⚠️", f"Borderline case - score is close to a decision boundary ({level} zone). Recommend human review rather than automated action.", True

    if level == "Low":
        return "Approve", "🟢", "Low risk profile - no action needed", False
    elif level == "Medium":
        return "Monitor", "🟡", "Some risk signals present - add to watchlist", False
    elif level == "High":
        return "Manual Review", "🟠", "Multiple risk signals present - needs analyst review", False
    else:
        return "Enhanced Investigation", "🔴", "Critical risk score - requires deep investigation", False

if __name__ == "__main__":
    test_cases = [(15, 0.02), (24, 0.02), (42, 0.03), (68, 0.05), (91, 0.02), (60, 0.15)]
    print("Testing Decision Engine with Borderline Detection")
    print("=" * 60)
    for score, cb_rate in test_cases:
        action, emoji, reason, is_border = get_recommended_action(score, cb_rate)
        print(f"Score: {score} | Chargeback: {cb_rate*100:.0f}% -> {emoji} {action}")
        print(f"  Borderline: {is_border} | Reason: {reason}\n")