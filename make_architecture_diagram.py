import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(6, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 16)
ax.axis('off')

stages = [
    ("Merchant Data", "2,000 synthetic profiles", "#5DCAA5"),
    ("XGBoost Model", "Risk score 0-100, 0.95 AUC", "#85B7EB"),
    ("SHAP Explainability", "Why each score was given", "#AFA9EC"),
    ("Streamlit Dashboard", "Interactive risk explorer", "#F0997B"),
    ("Live Deployment", "Public URL, no setup needed", "#97C459"),
]

y_positions = [14, 11, 8, 5, 2]

for (title, subtitle, color), y in zip(stages, y_positions):
    box = FancyBboxPatch((1, y - 0.7), 8, 1.4, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='black', linewidth=0.5)
    ax.add_patch(box)
    ax.text(5, y + 0.15, title, ha='center', va='center', fontsize=13, fontweight='bold')
    ax.text(5, y - 0.35, subtitle, ha='center', va='center', fontsize=10)
    if y != 2:
        ax.annotate('', xy=(5, y - 1.5), xytext=(5, y - 0.7),
                     arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

plt.title("MerchantGuard AI - Architecture", fontsize=15, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('architecture_diagram.png', dpi=150, bbox_inches='tight')
print("Saved architecture_diagram.png")