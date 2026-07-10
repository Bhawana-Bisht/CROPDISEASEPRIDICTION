import matplotlib.pyplot as plt

models = ["ResNet18","MobileNetV2","VGG16","EfficientNetB0"]
accuracy = [98.9,99.8,97.9,100.0]

colors = ["red","blue","green","purple"]

plt.figure(figsize=(8,5))
bars = plt.bar(models, accuracy, color=colors)

plt.ylabel("Accuracy (%)")
plt.xlabel("CNN Models")
plt.title("Training Accuracy Comparison Across CNN Architectures")

for bar in bars:
    plt.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.1,
        f"{bar.get_height():.1f}%",
        ha='center'
    )

plt.ylim(95,101)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()

plt.savefig("results/training_accuracy_comparison.png", dpi=300)
plt.show()