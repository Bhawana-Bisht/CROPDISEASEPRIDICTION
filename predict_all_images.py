import torch
from torchvision import transforms, models
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

# -------------------------------
# Settings
# -------------------------------
MODEL_PATH = "resnet18_rice_disease_model.pth"
MODEL_CHOICE = "resnet18"   # or "mobilenet_v3_small"
CLASS_MAPPING_PATH = "class_mapping.pth"
DATASET_FOLDER = r"data/rice_leaf_diseases"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙ Using device: {device}\n")

# -------------------------------
# FIXED: Load class mapping safely
# -------------------------------
try:
    idx_to_label = torch.load(CLASS_MAPPING_PATH, weights_only=False)
    classes = [idx_to_label[i] for i in range(len(idx_to_label))]
    print(f"✅ Class mapping loaded successfully! ({len(classes)} classes)\n")
except Exception as e:
    print("❌ ERROR loading class_mapping.pth:", e)
    exit()

num_classes = len(classes)

# -------------------------------
# Model initialization
# -------------------------------
print(f"🔄 Initializing model: {MODEL_CHOICE}...")

if MODEL_CHOICE == "resnet18":
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

elif MODEL_CHOICE == "mobilenet_v3_small":
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)

else:
    raise ValueError("Invalid MODEL_CHOICE selected")

# -------------------------------
# Load model weights safely
# -------------------------------
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
except:
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

# Remove mismatched layers
for key in ["fc.weight", "fc.bias", "classifier.3.weight", "classifier.3.bias"]:
    checkpoint.pop(key, None)

model.load_state_dict(checkpoint, strict=False)
model = model.to(device)
model.eval()

print("✅ Model Loaded Successfully!\n")

# -------------------------------
# Transforms
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------
# Collect all images
# -------------------------------
image_list = []
for root, _, files in os.walk(DATASET_FOLDER):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            image_list.append(os.path.join(root, file))

print(f"🔍 Total images found for prediction: {len(image_list)}")
print("⏳ Predicting... Please wait.\n")

# -------------------------------
# Prediction
# -------------------------------
results = []

for img_path in tqdm(image_list, desc="Predicting"):
    try:
        image = Image.open(img_path).convert("RGB")
        img_t = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_t)
            probabilities = torch.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probabilities, 1)

        results.append({
            "image_path": img_path.replace("\\", "/"),
            "predicted_class": idx_to_label[pred_idx.item()],
            "confidence": round(float(confidence.item()) * 100, 2)
        })

    except Exception as e:
        print(f"⚠ Error reading image: {img_path} | {e}")

# -------------------------------
# Save CSV
# -------------------------------
df = pd.DataFrame(results)
df.to_csv("predictions.csv", index=False)

print("\n📁 Results saved to: predictions.csv")
print(f"📊 Total images successfully predicted: {len(results)}")
print("🎉 Task Completed!")
