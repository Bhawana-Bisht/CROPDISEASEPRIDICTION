import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os

# -----------------------------
# Config: Updated paths
# -----------------------------
TRAIN_CSV = "data/splits/train.csv"
VAL_CSV   = "data/splits/val.csv"

DEVICE = "cpu"  # ya "cuda" agar GPU available ho
EPOCHS = 20
BATCH_SIZE = 16
NUM_CLASSES = 5  # total disease classes

# -----------------------------
# Dataset Class
# -----------------------------
class RiceDataset(Dataset):
    def __init__(self, csv_file):
        print(f"📂 Loading CSV: {csv_file}")
        self.df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(self.df)} rows")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = int(row["label"])

        # check if image exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"❌ Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label

# -----------------------------
# DataLoaders
# -----------------------------
train_dataset = RiceDataset(TRAIN_CSV)
val_dataset   = RiceDataset(VAL_CSV)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -----------------------------
# Model
# -----------------------------
model = models.vgg16(weights=None)
model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(EPOCHS):
    print(f"🚀 Epoch {epoch+1}/{EPOCHS} starting...")
    model.train()
    total_loss = 0

    for i, (x, y) in enumerate(train_loader, 1):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0 or i == len(train_loader):
            print(f"   Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1} Completed - Avg Train Loss: {avg_loss:.4f}")

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    val_acc = 100 * correct / total
    print(f"🔹 Validation Accuracy: {val_acc:.2f}%\n")

# -----------------------------
# Save Model
# -----------------------------
MODEL_SAVE_PATH = "models/vgg16_rice.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ VGG16 model saved as {MODEL_SAVE_PATH}")