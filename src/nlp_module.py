from keybert import KeyBERT
import json

# Load disease info from JSON
with open("disease_info.json", "r") as f:
    disease_info = json.load(f)

# KeyBERT model for symptom extraction
kw_model = KeyBERT()

def extract_symptoms(text):
    """
    Farmer ke symptoms text se keywords extract kare
    """
    keywords = kw_model.extract_keywords(text, top_n=5)
    return [kw[0] for kw in keywords]

def get_advice(disease):
    """
    Disease ke naam se easy language me advice return kare
    """
    info = disease_info.get(disease, {})
    advice_text = f"🌱 Disease Detected: {disease}\n"
    advice_text += f"📄 Description: {info.get('description', 'N/A')}\n"
    advice_text += f"💊 Treatment: {info.get('treatment', 'N/A')}\n"
    advice_text += f"🛡️ Prevention: {info.get('prevention', 'N/A')}"
    return advice_text