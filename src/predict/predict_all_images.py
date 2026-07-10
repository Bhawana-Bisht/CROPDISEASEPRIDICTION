import torch
from torchvision import transforms, models
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm
import argparse

# -------------------------------
# Arguments (optional CLI support)
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="data/rice_leaf_diseases", help="Folder with images to predict")
parser.add_argument("--model_path", type=str, default="models/resnet18_rice_disease_model.pth", help="Path to trained model")
parser.add_argument("--class_mapping", type=str, default="models/class_mapping.pth", help="Path to class mapping file")
parser.add_argument("--model_choice", type=str, default="resnet18", choices=["resnet18", "mobilenet_v3_small"], help="Model type")
args = parser.parse_args()

DATASET_FOLDER = args.dataset
MODEL_PATH = args.model_path
CLASS_MAPPING_PATH = args.class_mapping
MODEL_CHOICE = args.model_choice

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙ Using device: {device}\n")

# -------------------------------
# Load class mapping safely
# -------------------------------
try:
    # PyTorch 2.6+ fix: load full object safely
    idx_to_label = torch.load(CLASS_MAPPING_PATH, weights_only=False)
    classes = [idx_to_label[i] for i in range(len(idx_to_label))]
    print(f"✅ Class mapping loaded successfully! ({len(classes)} classes)\n")
except Exception as e:
    print("❌ ERROR loading class mapping:", e)
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
# Load model weights
# -------------------------------
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint, strict=False)
model = model.to(device)
model.eval()
print("✅ Model Loaded Successfully!\n")

# -------------------------------
# Image transforms
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
# Save results to CSV
# -------------------------------
df = pd.DataFrame(results)
df.to_csv("predictions.csv", index=False)

print("\n📁 Results saved to: predictions.csv")
print(f"📊 Total images successfully predicted: {len(results)}")
print("🎉 Task Completed!")