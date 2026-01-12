import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
import json
import os

# ----------------------------
# CONFIG
# ----------------------------
TEST_DIR = "data/test"
MODEL_PATH = "rice_disease_model.pth"
NUM_CLASSES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# DATA
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder(TEST_DIR, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# ----------------------------
# MODEL
# ----------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ----------------------------
# EVALUATION
# ----------------------------
y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        preds = torch.argmax(probs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

# ----------------------------
# METRICS
# ----------------------------
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted"
)

auc = roc_auc_score(
    np.eye(NUM_CLASSES)[y_true],
    np.array(y_prob),
    multi_class="ovr"
)

# ----------------------------
# SAVE METRICS
# ----------------------------
os.makedirs("results", exist_ok=True)

metrics = {
    "cnn_models": {
        "ResNet18": {
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1": round(f1 * 100, 2),
            "auc": round(auc, 3)
        }
    }
}

with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Real evaluation completed and metrics saved.")
