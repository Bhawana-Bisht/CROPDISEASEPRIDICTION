import json
import matplotlib.pyplot as plt

with open("results/training_history.json","r") as f:
    history = json.load(f)

epochs = range(1, len(history["train_loss"])+1)

# Accuracy Curve
plt.figure(figsize=(8,5))
plt.plot(epochs, history["train_accuracy"],
         marker='o', color='green',
         label='Training Accuracy')

plt.plot(epochs, history["val_accuracy"],
         marker='s', color='red',
         label='Validation Accuracy')

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("results/training_validation_accuracy_curve.png")
plt.close()

# Loss Curve
plt.figure(figsize=(8,5))
plt.plot(epochs, history["train_loss"],
         marker='o', color='blue',
         label='Training Loss')

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("results/training_loss_curve.png")
plt.close()

print("Graphs Saved")