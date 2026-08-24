import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(6, 18))
ax.set_xlim(0, 10)
ax.set_ylim(0, 29)
ax.axis('off')

stages = [
    ("Merchant Data", "2,000 synthetic merchant profiles", "#E8F1FB"),
    ("Data Validation & Feature Processing", "Transaction, refund, chargeback,\naccount-age & growth signals", "#F1F5F9"),
    ("XGBoost Chargeback-Risk Model", "Risk score 0-100", "#DCE8F5"),
    ("SHAP Explainability", "Why was this merchant flagged?", "#E9E3F5"),
    ("Decision Engine", "Approve / Monitor / Manual Review /\nEnhanced Investigation", "#FFF0E6"),
    ("AI Investigation Report", "Risk summary + evidence +\nrecommended action", "#FFF7E6"),
    ("Human Analyst Decision", "Accept recommendation or\noverride with reason", "#E8F5ED"),
    ("Audit Trail", "Final decision + reason + timestamp", "#EAF4F4"),
    ("Streamlit Dashboard & Live Deployment", "Interactive risk management system", "#DCE8F5"),
]

text_color = "#1B2A4A"
n = len(stages)
y_positions = [26 - i * 3 for i in range(n)]

for (title, subtitle, color), y in zip(stages, y_positions):
    box = FancyBboxPatch((1, y - 1.1), 8, 2.2, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor=text_color, linewidth=0.8)
    ax.add_patch(box)
    ax.text(5, y + 0.35, title, ha='center', va='center',
            fontsize=12, fontweight='bold', color=text_color)
    ax.text(5, y - 0.45, subtitle, ha='center', va='center',
            fontsize=9, color=text_color)
    if y != y_positions[-1]:
        ax.annotate('', xy=(5, y - 1.9), xytext=(5, y - 1.1),
                     arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

plt.title("MerchantGuard AI - Architecture", fontsize=16, fontweight='bold',
          color=text_color, pad=20)
plt.tight_layout()
plt.savefig('architecture_diagram.png', dpi=150, bbox_inches='tight')
print("Saved architecture_diagram.png")