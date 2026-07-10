import matplotlib.pyplot as plt

models = ["ResNet18","MobileNetV2","VGG16","EfficientNetB0"]

validation_loss = [
    0.045,
    0.021,
    0.067,
    0.010
]

colors = ["orange","cyan","lime","magenta"]

plt.figure(figsize=(8,5))
bars = plt.bar(models, validation_loss, color=colors)

plt.ylabel("Loss")
plt.xlabel("CNN Models")
plt.title("Validation Loss Comparison Across CNN Architectures")

for bar in bars:
    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.002,
        f"{bar.get_height():.3f}",
        ha='center'
    )

plt.grid(axis='y', linestyle='--')
plt.tight_layout()

plt.savefig("results/validation_loss_comparison.png", dpi=300)
plt.show()