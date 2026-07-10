import json
import matplotlib.pyplot as plt

# Load metrics
with open("results/per_class_metrics.json", "r") as f:
    data = json.load(f)

classes = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4"]

precision = [data[str(i)]["precision"] * 100 for i in range(5)]
recall    = [data[str(i)]["recall"] * 100 for i in range(5)]
f1_score  = [data[str(i)]["f1-score"] * 100 for i in range(5)]

# IEEE-style settings
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10
})

# -----------------------------
# Precision Graph
# -----------------------------
plt.figure(figsize=(8,5))

bars = plt.bar(
    classes,
    precision,
    color="#1f77b4",
    edgecolor="black",
    linewidth=1
)

plt.title("Per-Class Precision")
plt.ylabel("Precision (%)")
plt.ylim(0, 110)
plt.grid(axis="y", linestyle="--", alpha=0.5)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 1,
        f"{bar.get_height():.1f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig("results/per_class_precision.png", dpi=300)
plt.close()

# -----------------------------
# Recall Graph
# -----------------------------
plt.figure(figsize=(8,5))

bars = plt.bar(
    classes,
    recall,
    color="#ff7f0e",
    edgecolor="black",
    linewidth=1
)

plt.title("Per-Class Recall")
plt.ylabel("Recall (%)")
plt.ylim(0, 110)
plt.grid(axis="y", linestyle="--", alpha=0.5)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 1,
        f"{bar.get_height():.1f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig("results/per_class_recall.png", dpi=300)
plt.close()

# -----------------------------
# F1 Score Graph
# -----------------------------
plt.figure(figsize=(8,5))

bars = plt.bar(
    classes,
    f1_score,
    color="#2ca02c",
    edgecolor="black",
    linewidth=1
)

plt.title("Per-Class F1-Score")
plt.ylabel("F1 Score (%)")
plt.ylim(0, 110)
plt.grid(axis="y", linestyle="--", alpha=0.5)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 1,
        f"{bar.get_height():.1f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig("results/per_class_f1.png", dpi=300)
plt.close()

print("IEEE-style graphs saved successfully!")