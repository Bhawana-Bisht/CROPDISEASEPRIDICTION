import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os

# -----------------------------
# Config (UPDATED PATHS)
# -----------------------------
TRAIN_CSV = "data/splits/train.csv"
VAL_CSV   = "data/splits/val.csv"
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Dataset Class
# -----------------------------
class RiceDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["image_path"]
        label = int(row["label"])

        # 🔥 FIX: ensure correct path (important after moving folders)
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.getcwd(), img_path)

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        return img, label


# -----------------------------
# Load Data
# -----------------------------
train_data = RiceDataset(TRAIN_CSV)
val_data   = RiceDataset(VAL_CSV)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Model
# -----------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 5)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)

        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")

    # -----------------------------
    # Validation Accuracy
    # -----------------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            out = model(x)
            pred = out.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = correct / total * 100
    print(f"Validation Accuracy: {acc:.2f}%")


# -----------------------------
# Save model (UPDATED PATH)
# -----------------------------
MODEL_PATH = "models/resnet18_rice.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Model saved as {MODEL_PATH}")