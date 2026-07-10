import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# -----------------------------
# Fix Paths
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

train_csv = os.path.join(BASE_DIR, "data/splits/train.csv")
val_csv   = os.path.join(BASE_DIR, "data/splits/val.csv")
test_csv  = os.path.join(BASE_DIR, "data/splits/test.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("📂 Base Dir:", BASE_DIR)

# -----------------------------
# Fixed Class Names
# -----------------------------
CLASS_NAMES = ["Bacterialblight", "Blast", "Brownspot", "HealthyLeaf", "Tungro"]
class_to_idx = {cls: i for i, cls in enumerate(CLASS_NAMES)}
num_classes = len(CLASS_NAMES)

print("✅ Class Mapping:", class_to_idx)

# -----------------------------
# Dataset class
# -----------------------------
class RiceLeafDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(BASE_DIR, row['image_path'])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"❌ Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        # 🔥 IMPORTANT FIX: directly use label (no remapping confusion)
        label = int(row['label'])

        if self.transform:
            image = self.transform(image)

        return image, label

# -----------------------------
# Load CSVs
# -----------------------------
train_df = pd.read_csv(train_csv)
val_df   = pd.read_csv(val_csv)
test_df  = pd.read_csv(test_csv)

# -----------------------------
# Class Weights
# -----------------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(num_classes),
    y=train_df['label'].values
)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# -----------------------------
# Transforms (FIXED)
# -----------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# Datasets & Loaders
# -----------------------------
train_dataset = RiceLeafDataset(train_df, transform=train_transforms)
val_dataset   = RiceLeafDataset(val_df, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙ Using device: {device}")

# -----------------------------
# Model (🔥 FIX: Freeze Layers)
# -----------------------------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)
print("🔄 Model initialized: ResNet18 (Frozen)")

# -----------------------------
# Loss & Optimizer (🔥 FIX)
# -----------------------------
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# Only train last layer
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

# -----------------------------
# Training
# -----------------------------
num_epochs = 15   # 🔥 Increased epochs

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

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

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

# -----------------------------
# Save Model
# -----------------------------
model_path = os.path.join(MODEL_DIR, "resnet18_rice_model.pth")
torch.save(model.state_dict(), model_path)

class_path = os.path.join(MODEL_DIR, "class_names.pth")
torch.save(CLASS_NAMES, class_path)

print("✅ Model Saved:", model_path)
print("✅ Class Names Saved:", class_path)