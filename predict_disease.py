import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse
import os

# ---------------------------------
# Arguments
# ---------------------------------
parser = argparse.ArgumentParser(description='Predict rice leaf disease')
parser.add_argument('--image', type=str, required=True, help='Path to the image')
args = parser.parse_args()

# ---------------------------------
# Device
# ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙ Using device: {device}")

# ---------------------------------
# Fixed Class Labels
# ---------------------------------
classes = ["Bacterialblight", "Blast", "Brownspot", "HealthyLeaf", "Tungro"]
print(f"✅ Class labels loaded ({len(classes)} classes)")

# ---------------------------------
# Load Model
# ---------------------------------
MODEL_PATH = "rice_disease_model.pth"

num_classes = len(classes)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded successfully: {MODEL_PATH}")
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# ---------------------------------
# Image Preprocessing
# ---------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------------------------
# Load Image
# ---------------------------------
if not os.path.exists(args.image):
    raise FileNotFoundError(f"Image not found: {args.image}")

image = Image.open(args.image).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# ---------------------------------
# Prediction
# ---------------------------------
with torch.no_grad():
    outputs = model(image)
    probabilities = torch.softmax(outputs, dim=1)
    predicted_idx = torch.argmax(probabilities, dim=1).item()
    predicted_class = classes[predicted_idx]
    confidence = probabilities[0][predicted_idx].item() * 100

# ---------------------------------
# CNN Output
# ---------------------------------
print("\n⭐ Predicted Class:", predicted_class)
print(f"📊 Confidence: {confidence:.2f}%")

# ---------------------------------
# LLM BASED EXPLANATION
# ---------------------------------
from llm_module import generate_llm_response

llm_output = generate_llm_response(predicted_class)
print(llm_output)
