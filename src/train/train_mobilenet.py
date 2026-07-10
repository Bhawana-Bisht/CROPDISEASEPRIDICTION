import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# ==========================================================
#       Custom Dataset (Automatically maps class → index)
# ==========================================================
class RiceDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"❌ CSV file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # Detect unique classes
        classes = sorted(self.df["label"].unique().tolist())
        print("Detected classes:", classes)

        # Mapping class → index
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        print("Class to index mapping:", self.class_to_idx)

        self.labels = self.df["label"].values
        self.image_paths = self.df["image_path"].values

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"❌ Image not found: {img_path}")

        label = self.class_to_idx[label]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.df)

# ==========================================================
#                 CONFIGURATION
# ==========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", DEVICE)

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4

# ==========================================================
#                 TRANSFORMS
# ==========================================================
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==========================================================
#                 LOAD DATA
# ==========================================================
train_loader = DataLoader(
    RiceDataset("data/splits/train.csv", train_tf),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    RiceDataset("data/splits/val.csv", val_tf),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================================================
#                 MODEL: MobileNetV2
# ==========================================================
NUM_CLASSES = len(train_loader.dataset.class_to_idx)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ==========================================================
#                 TRAINING LOOP
# ==========================================================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        _, predicted = torch.max(out.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    train_loss = running_loss / total
    train_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}%")

# ==========================================================
#                 VALIDATION
# ==========================================================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

accuracy = correct / total * 100
print(f"Validation Accuracy: {accuracy:.2f}%")

# ==========================================================
#                 SAVE MODEL & CLASS MAPPING
# ==========================================================
os.makedirs("models", exist_ok=True)
MODEL_SAVE_PATH = "models/mobilenet_v2_rice.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Model saved as {MODEL_SAVE_PATH}")

# Save class mapping for prediction scripts
class_mapping = train_loader.dataset.class_to_idx
torch.save(class_mapping, "models/class_mapping.pth")
print("✅ Class mapping saved as models/class_mapping.pth")