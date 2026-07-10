import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import argparse
import sys

# ---------------------------------
# Fix Base Directory
# ---------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../.."))

sys.path.append(BASE_DIR)

# ---------------------------------
# Paths
# ---------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "models", "resnet18_rice_model.pth")
CLASS_PATH = os.path.join(BASE_DIR, "models", "class_names.pth")

# ---------------------------------
# Device
# ---------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙ Using device: {device}")

# ---------------------------------
# Load Class Names
# ---------------------------------
if os.path.exists(CLASS_PATH):
    classes = torch.load(CLASS_PATH, map_location=device)
    print(f"✅ Classes loaded: {classes}")
else:
    raise FileNotFoundError(f"❌ Class file not found: {CLASS_PATH}")

num_classes = len(classes)

# ---------------------------------
# Load Model
# ---------------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded: {MODEL_PATH}")
else:
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

# ---------------------------------
# Transform
# ---------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------------
# LLM Module
# ---------------------------------
from llm_module import generate_llm_response


# ---------------------------------
# Prediction Function
# ---------------------------------
def predict_disease(image_path, user_query="", language="Hindi"):

    if user_query == "":
        try:
            user_query = input("👨‍🌾 Farmer: Apni problem batayein: ")
        except Exception:
            user_query = ""

    # ---------------------------------
    # Image Path Fix
    # ---------------------------------
    if not os.path.isabs(image_path):
        image_path = os.path.join(BASE_DIR, image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

    # ---------------------------------
    # Image Processing
    # ---------------------------------
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    # ---------------------------------
    # Model Prediction
    # ---------------------------------
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

        predicted_idx = torch.argmax(probs, dim=1).item()
        predicted_class = classes[predicted_idx]
        confidence = probs[0][predicted_idx].item() * 100

    # ---------------------------------
    # Smart Advice System
    # ---------------------------------
    try:

        disease = predicted_class.lower()
        query = user_query.lower()

        # =========================================
        # 🌿 HEALTHY LEAF
        # =========================================
        if disease == "healthyleaf":

            if language == "Hindi":

                llm_output = """
🌿 आपकी फसल स्वस्थ है।

✔ नियमित पानी और संतुलित खाद बनाए रखें।

🛡 बचाव:
- मिट्टी की जांच करवाएं
- ज्यादा पानी भरने से बचें
- नियमित निगरानी करें
"""

            elif language == "English":

                llm_output = """
🌿 Your crop is healthy.

✔ Maintain proper irrigation and balanced fertilizer.

🛡 Prevention:
- Perform soil testing
- Avoid overwatering
- Monitor crop regularly
"""

            else:  # Hinglish

                llm_output = """
🌿 Aapki fasal healthy hai.

✔ Regular paani aur balanced khaad maintain rakhein.

🛡 Prevention:
- Soil testing karwayein
- Overwatering se bachein
- Regular monitoring karein
"""

        # =========================================
        # 🦠 BACTERIAL BLIGHT
        # =========================================
        elif disease == "bacterialblight":

            if language == "Hindi":

                llm_output = """
🦠 बैक्टीरियल ब्लाइट बीमारी पाई गई है।

🔍 लक्षण:
- पत्तियों का पीला होना
- किनारों से सूखना

💊 उपचार:
- स्ट्रेप्टोमाइसिन स्प्रे करें
- कॉपर ऑक्सीक्लोराइड का उपयोग करें

🛡 बचाव:
- खेत में अधिक पानी न भरें
- रोग प्रतिरोधी किस्म लगाएं
"""

            elif language == "English":

                llm_output = """
🦠 Bacterial Blight disease detected.

🔍 Symptoms:
- Yellow leaves
- Drying from leaf edges

💊 Treatment:
- Spray Streptomycin
- Use Copper Oxychloride

🛡 Prevention:
- Avoid water stagnation
- Use resistant varieties
"""

            else:  # Hinglish

                llm_output = """
🦠 Bacterial Blight detect hua hai.

🔍 Symptoms:
- Patton ka yellow hona
- Leaf edges se sukna

💊 Treatment:
- Streptomycin spray karein
- Copper oxychloride use karein

🛡 Prevention:
- Zyada paani na bharein
- Resistant variety use karein
"""

        # =========================================
        # 🍂 BROWNSPOT
        # =========================================
        elif disease == "brownspot":

            if language == "Hindi":

                llm_output = """
🍂 ब्राउनस्पॉट बीमारी पाई गई है।

🔍 लक्षण:
- पत्तियों पर भूरे धब्बे
- पत्तियों का सूखना

💊 उपचार:
- मैंकोज़ेब स्प्रे करें

🛡 बचाव:
- खेत में उचित जल निकासी रखें
"""

            elif language == "English":

                llm_output = """
🍂 Brownspot disease detected.

🔍 Symptoms:
- Brown spots on leaves
- Leaf drying

💊 Treatment:
- Spray Mancozeb

🛡 Prevention:
- Maintain proper drainage
"""

            else:  # Hinglish

                llm_output = """
🍂 Brownspot disease detect hui hai.

🔍 Symptoms:
- Patton par brown spots
- Leaf drying

💊 Treatment:
- Mancozeb spray karein

🛡 Prevention:
- Proper drainage rakhein
"""

        # =========================================
        # 🌾 BLAST
        # =========================================
        elif disease == "blast":

            if language == "Hindi":

                llm_output = """
🌾 ब्लास्ट बीमारी पाई गई है।

🔍 लक्षण:
- हीरे जैसे धब्बे
- पत्तियों का जलना

💊 उपचार:
- ट्राइसाइक्लाजोल स्प्रे करें

🛡 बचाव:
- संतुलित खाद का उपयोग करें
"""

            elif language == "English":

                llm_output = """
🌾 Blast disease detected.

🔍 Symptoms:
- Diamond-shaped spots
- Leaf burning

💊 Treatment:
- Spray Tricyclazole

🛡 Prevention:
- Use balanced fertilizer
"""

            else:  # Hinglish

                llm_output = """
🌾 Blast disease detect hua hai.

🔍 Symptoms:
- Diamond spots
- Leaf burning

💊 Treatment:
- Tricyclazole spray karein

🛡 Prevention:
- Balanced fertilizer use karein
"""

        # =========================================
        # 🐛 TUNGRO
        # =========================================
        elif disease == "tungro":

            if language == "Hindi":

                llm_output = """
🐛 टंग्रो वायरस पाया गया है।

🔍 लक्षण:
- पत्तियों का पीला होना
- पौधे की वृद्धि रुकना

💊 उपचार:
- इमिडाक्लोप्रिड स्प्रे करें

🛡 बचाव:
- रोग प्रतिरोधी किस्म लगाएं
"""

            elif language == "English":

                llm_output = """
🐛 Tungro virus detected.

🔍 Symptoms:
- Yellow leaves
- Stunted growth

💊 Treatment:
- Spray Imidacloprid

🛡 Prevention:
- Use resistant varieties
"""

            else:  # Hinglish

                llm_output = """
🐛 Tungro virus detect hua hai.

🔍 Symptoms:
- Yellow leaves
- Growth ruk jana

💊 Treatment:
- Imidacloprid spray karein

🛡 Prevention:
- Resistant variety use karein
"""

        # =========================================
        # OTHER DISEASES
        # =========================================
        else:
            llm_output = generate_llm_response(
                predicted_class,
                user_query
            )

    except Exception as e:

        print("❌ ERROR:", e)

        llm_output = "⚠ Advice generate nahi ho paayi."

    # ---------------------------------
    # Final Result
    # ---------------------------------
    result = {
        "disease": predicted_class,
        "confidence": round(confidence, 2),
        "llm_advice": llm_output
    }

    # ---------------------------------
    # Console Output
    # ---------------------------------
    print("\n🌿 Prediction Result")
    print("⭐ Disease:", predicted_class)
    print(f"📊 Confidence: {confidence:.2f}%")

    print("\n🤖 AI Advisor:")
    print(llm_output)

    return result


# ---------------------------------
# CLI Testing
# ---------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        required=False
    )

    parser.add_argument(
        "--query",
        type=str,
        default=""
    )

    parser.add_argument(
        "--language",
        type=str,
        default="Hindi"
    )

    args = parser.parse_args()

    test_image = (
        args.image or
        "data/rice_leaf_diseases/Brownspot/brownspot_orig_010.jpg"
    )

    user_query = args.query or ""
    language = args.language or "Hindi"

    result = predict_disease(
        test_image,
        user_query,
        language
    )

    print("\n✅ Final Output:", result)
