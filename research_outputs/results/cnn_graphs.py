import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

# -----------------------------
# CNN Accuracy (example values)
# -----------------------------
epochs = [1,2,3,4,5,6,7,8,9,10]
train_acc = [70,75,80,85,88,90,92,94,95,96]
val_acc   = [68,74,78,82,86,88,90,91,92,93]

plt.figure(figsize=(6,4))
plt.plot(epochs, train_acc, label="Training Accuracy")
plt.plot(epochs, val_acc, label="Validation Accuracy")

plt.title("CNN Model Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid()

plt.savefig("results/cnn_accuracy_ieee.png", dpi=300)
plt.show()