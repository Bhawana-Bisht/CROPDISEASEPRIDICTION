import matplotlib.pyplot as plt

epochs = list(range(1, 11))

train_acc = [
    87.46,
    99.24,
    99.79,
    99.83,
    99.87,
    99.81,
    99.92,
    99.89,
    99.73,
    99.94
]

val_acc = [
    100.00,
    100.00,
    100.00,
    100.00,
    100.00,
    100.00,
    99.91,
    100.00,
    100.00,
    100.00
]

plt.figure(figsize=(8,5))

plt.plot(
    epochs,
    train_acc,
    marker='o',
    linewidth=2,
    label='Training Accuracy'
)

plt.plot(
    epochs,
    val_acc,
    marker='s',
    linewidth=2,
    label='Validation Accuracy'
)

plt.title(
    "Training and Validation Accuracy Curves for EfficientNet-B0",
    fontsize=12,
    fontweight='bold'
)

plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/training_validation_accuracy_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Graph saved successfully!")