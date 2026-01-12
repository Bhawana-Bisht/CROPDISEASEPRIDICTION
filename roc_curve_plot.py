import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "rice_disease_model.pth"
DATASET_PATH = "data/rice_leaf_diseases"
NUM_CLASSES = 5
CLASS_NAMES = ["Bacterialblight", "Blast", "Brownspot", "HealthyLeaf", "Tungro"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# Dataset & Loader
# -----------------------------
dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

# -----------------------------
# Load Model
# -----------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# -----------------------------
# Collect Predictions
# -----------------------------
y_true = []
y_scores = []

with torch.no_grad():
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        y_scores.append(probs.cpu().numpy())
        y_true.append(labels.numpy())

y_true = np.concatenate(y_true)
y_scores = np.concatenate(y_scores)

# -----------------------------
# Binarize Labels
# -----------------------------
y_true_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

# -----------------------------
# Plot ROC Curve
# -----------------------------
plt.figure(figsize=(8, 6))

for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC Curve for Rice Leaf Disease Classification")
plt.legend(loc="lower right")
plt.grid(True)

os.makedirs("results/roc_curves", exist_ok=True)
plt.savefig("results/roc_curves/roc_curve.png", dpi=300)
plt.show()
