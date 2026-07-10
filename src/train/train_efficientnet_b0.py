import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import json

# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", DEVICE)

# -----------------------------
# Dataset Class with Class Mapping
# -----------------------------
class RiceDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"❌ CSV not found: {csv_file}")

        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # Get unique classes and mapping
        classes = sorted(self.data['label'].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        print("Detected classes:", classes)
        print("Class to index mapping:", self.class_to_idx)

        self.image_paths = self.data['image_path'].values
        self.labels = [self.class_to_idx[lbl] for lbl in self.data['label'].values]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"❌ Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# -----------------------------
# Transforms
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# Load Data
# -----------------------------
train_dataset = RiceDataset("data/splits/train.csv", train_transform)
val_dataset = RiceDataset("data/splits/val.csv", val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

NUM_CLASSES = len(train_dataset.class_to_idx)

# -----------------------------
# Load EfficientNet-B0
# -----------------------------
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

# -----------------------------
# Loss & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -----------------------------
# Training History
# -----------------------------
history = {
    "train_loss": [],
    "train_acc": [],
    "val_acc": []
}


# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / total
    train_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}%")
        # Save metrics for graphs
    history["train_loss"].append(float(train_loss))
    history["train_acc"].append(float(train_acc))
    history["val_acc"].append(float(val_acc))

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.2f}%")

# -----------------------------
# Save Model & Class Mapping
# -----------------------------
os.makedirs("models", exist_ok=True)
MODEL_SAVE_PATH = "models/efficientnet_b0_rice.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ EfficientNet-B0 model saved as {MODEL_SAVE_PATH}")

# Save class mapping
torch.save(train_dataset.class_to_idx, "models/class_mapping.pth")
print("✅ Class mapping saved as models/class_mapping.pth")
# -----------------------------
# Save Training History
# -----------------------------
os.makedirs("results", exist_ok=True)

with open("results/training_history.json", "w") as f:
    json.dump(history, f, indent=4)

print("✅ Training history saved to results/training_history.json")