import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image
import os

# Step 1: Check GPU / CUDA
print("✅ Torch version:", torch.__version__)
print("✅ Torchvision version:", torchvision.__version__)
print("✅ CUDA available:", torch.cuda.is_available())

# Step 2: Check if dataset folder exists
data_path = "data/rice-plant-diseases-dataset"
if os.path.exists(data_path):
    print(f"✅ Dataset folder found at: {data_path}")
    print("📁 Classes:", os.listdir(data_path))
else:
    print("❌ Dataset folder not found! Please check path.")

# Step 3: Try loading a small batch using ImageFolder
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

try:
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    print(f"✅ Dataset loaded successfully with {len(dataset)} images.")
except Exception as e:
    print("❌ Error loading dataset:", e)
