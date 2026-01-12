import os
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# -----------------------------
# Dataset class
# -----------------------------
class RiceLeafDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = self.df['label'].values
        self.image_paths = self.df['image_path'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# -----------------------------
# Paths
# -----------------------------
train_csv = "train_dataset.csv"
val_csv = "val_dataset.csv"
test_csv = "test_dataset.csv"

# -----------------------------
# Label encoding
# -----------------------------
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

le = LabelEncoder()
train_df['label'] = le.fit_transform(train_df['label'])
val_df['label'] = le.transform(val_df['label'])
test_df['label'] = le.transform(test_df['label'])

train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)
test_df.to_csv(test_csv, index=False)

num_classes = len(le.classes_)
print(f"✅ Classes: {le.classes_}")

# -----------------------------
# Transforms
# -----------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -----------------------------
# Datasets & Dataloaders
# -----------------------------
train_dataset = RiceLeafDataset(train_csv, transform=train_transforms)
val_dataset = RiceLeafDataset(val_csv, transform=val_transforms)
test_dataset = RiceLeafDataset(test_csv, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙ Using device: {device}")

# -----------------------------
# Model Selection: 'resnet18' or 'mobilenet_v3_small'
# -----------------------------
MODEL_CHOICE = "resnet18"  # Change to "mobilenet_v3_small" for MobileNetV3

if MODEL_CHOICE == "resnet18":
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
elif MODEL_CHOICE == "mobilenet_v3_small":
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
else:
    raise ValueError("Invalid MODEL_CHOICE. Choose 'resnet18' or 'mobilenet_v3_small'")

model = model.to(device)
print(f"🔄 Model initialized: {MODEL_CHOICE}")

# -----------------------------
# Loss & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# Training loop
# -----------------------------
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
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
        _, predicted = torch.max(outputs.data, 1)
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
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} "
          f"Train Acc: {train_acc:.2f}% "
          f"Val Acc: {val_acc:.2f}%")

# -----------------------------
# Save model & class mapping
# -----------------------------
MODEL_SAVE_PATH = f"{MODEL_CHOICE}_rice_disease_model.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Model saved as {MODEL_SAVE_PATH}")

# Save class mapping for prediction scripts
class_mapping = {i: cls for i, cls in enumerate(le.classes_)}
torch.save(class_mapping, "class_mapping.pth")
print("✅ Class mapping saved as class_mapping.pth")
