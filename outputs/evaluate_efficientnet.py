import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import classification_report

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/efficientnet_b0_rice.pth"
TEST_CSV = "data/splits/test.csv"
BATCH_SIZE = 32

class RiceDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        classes = sorted(self.data['label'].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        self.image_paths = self.data['image_path'].values
        self.labels = [self.class_to_idx[lbl] for lbl in self.data['label'].values]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_dataset = RiceDataset(TEST_CSV, transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

NUM_CLASSES = len(test_dataset.class_to_idx)

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())
        y_prob.extend(probs.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted"
)

auc = roc_auc_score(
    np.eye(NUM_CLASSES)[y_true],
    np.array(y_prob),
    multi_class="ovr"
)
# -----------------------------
# Per-Class Metrics
# -----------------------------
report = classification_report(
    y_true,
    y_pred,
    output_dict=True
)

print("\n===== Per-Class Metrics =====")

for cls in range(NUM_CLASSES):
    print(
        f"Class {cls}: "
        f"Precision={report[str(cls)]['precision']:.4f}, "
        f"Recall={report[str(cls)]['recall']:.4f}, "
        f"F1={report[str(cls)]['f1-score']:.4f}"
    )

# Save Per-Class Metrics
with open("results/per_class_metrics.json", "w") as f:
    json.dump(report, f, indent=4)

print("\nPer-class metrics saved to results/per_class_metrics.json")

print("\\n===== EfficientNet-B0 Results =====")
print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")
print(f"AUC       : {auc:.4f}")

os.makedirs("results", exist_ok=True)

metrics = {
    "cnn_models": {
        "EfficientNetB0": {
            "accuracy": round(accuracy*100, 2),
            "precision": round(precision*100, 2),
            "recall": round(recall*100, 2),
            "f1": round(f1*100, 2),
            "auc": round(auc, 4)
        }
    }
}

with open("results/efficientnet_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\\nMetrics saved to results/efficientnet_metrics.json")
