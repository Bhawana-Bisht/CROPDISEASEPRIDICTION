import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd

TRAIN_CSV = "train_dataset.csv"
VAL_CSV = "val_dataset.csv"

DEVICE = "cpu"
EPOCHS = 20
BATCH_SIZE = 16

class RiceDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row["image_path"]
        label = int(row["label"])

        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        return img, label


train_loader = DataLoader(RiceDataset(TRAIN_CSV), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(RiceDataset(VAL_CSV), batch_size=BATCH_SIZE)

# Load VGG16
model = models.vgg16(weights=None)
model.classifier[6] = nn.Linear(4096, 5)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        pred = model(x)

        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {total_loss/len(train_loader):.4f}")

    # validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            pred = model(x).argmax(1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    print(f"Validation Accuracy: {100*correct/total:.2f}%")

torch.save(model.state_dict(), "vgg16_rice.pth")
print("✅ VGG16 model saved as vgg16_rice.pth")
