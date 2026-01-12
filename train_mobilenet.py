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
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # Get unique labels like:
        # ['Bacterialblight','Blast','Brownspot','HealthyLeaf','Tungro']
        classes = sorted(self.df["label"].unique().tolist())
        print("Detected classes:", classes)

        # Create mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        print("Class to index mapping:", self.class_to_idx)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["image_path"]
        label = row["label"]
        label = self.class_to_idx[label]     # Convert label → index

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.df)

# ==========================================================
#                 CONFIGURATION
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Training on:", DEVICE)

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
NUM_CLASSES = 5

# ==========================================================
#                 TRANSFORMS
# ==========================================================
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
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
    RiceDataset("train_dataset.csv", train_tf),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    RiceDataset("val_dataset.csv", val_tf),
    batch_size=BATCH_SIZE
)

# ==========================================================
#                 MODEL: MobileNetV2
# ==========================================================
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

    for batch in train_loader:
        x, y = batch
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f}")

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
#                 SAVE MODEL
# ==========================================================
torch.save(model.state_dict(), "mobilenet_v2_rice.pth")
print("Model saved as mobilenet_v2_rice.pth")
