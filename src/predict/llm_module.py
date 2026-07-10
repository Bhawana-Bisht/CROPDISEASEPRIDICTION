import json
import os

# -----------------------------
# Paths (FIXED)
# -----------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(CURRENT_DIR, "disease_info.json")

# -----------------------------
# Try OpenAI Import (Safe)
# -----------------------------
USE_LLM = False
try:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        client = OpenAI(api_key=api_key)
        USE_LLM = True
except Exception as e:
    print("⚠ OpenAI not available, using JSON fallback:", e)
    USE_LLM = False


# -----------------------------
# LANGUAGE TRANSLATOR (SIMPLE RULE BASED)
# -----------------------------
def translate_text(text, language):
    if language == "English":
        return text

    if language == "Hindi":
        return text

    if language == "Hinglish":
        return text.replace("आपकी फसल", "Aapki fasal")

    return text


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def generate_llm_response(disease_name, user_input=None, language="Hindi"):

    original_name = disease_name
    disease_name = disease_name.strip().lower()

    # -----------------------------
    # 🟢 HEALTHY LEAF
    # -----------------------------
    if disease_name == "healthyleaf":

        if language == "English":
            response = "🌿 Your crop is healthy.\n"

            if user_input:
                response += f"\nFarmer input:\n{user_input}\n"

            response += """
✔ Maintain regular watering
✔ Use balanced fertilizer

🛡 Prevention:
- Soil testing recommended
- Avoid overwatering
"""
            return response

        elif language == "Hinglish":
            response = "🌿 Aapki fasal healthy hai.\n"

            if user_input:
                response += f"\nAapka input:\n{user_input}\n"

            response += """
✔ Regular paani aur khaad use karein

🛡 Prevention:
- Soil testing karo
- Overwatering avoid karo
"""
            return response

        else:
            response = "🌿 आपकी फसल स्वस्थ है।\n"

            if user_input:
                response += f"\n🧑‍🌾 आपने बताया:\n{user_input}\n"

            response += """
✔ नियमित पानी दें
✔ संतुलित खाद उपयोग करें

🛡 रोकथाम:
- मिट्टी परीक्षण कराएं
- ज्यादा पानी से बचें
"""
            return response


    # -----------------------------
    # 1️⃣ LLM MODE (OpenAI)
    # -----------------------------
    if USE_LLM and user_input:
        try:
            prompt = f"""
You are an agriculture expert.

Disease: {original_name}
Language: {language}

Farmer input:
"{user_input}"

Give:
- Simple explanation
- Symptoms
- Treatment
- Prevention

Make it farmer friendly.
"""
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content

        except Exception as e:
            print("⚠ LLM failed, switching to JSON:", e)

    # -----------------------------
    # 2️⃣ JSON FALLBACK
    # -----------------------------
    try:
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("❌ JSON LOAD ERROR:", e)
        data = {}

    data = {k.lower(): v for k, v in data.items()}

    info = data.get(disease_name, {
        "description": "No description available",
        "treatment": "No treatment available",
        "prevention": "No prevention available"
    })

    # -----------------------------
    # FINAL RESPONSE FORMAT
    # -----------------------------
    if language == "English":
        response = f"""
🌾 Disease: {original_name}

📌 Description:
{info['description']}

💊 Treatment:
{info['treatment']}

🛡 Prevention:
{info['prevention']}
"""

    elif language == "Hinglish":
        response = f"""
🌾 Disease: {original_name}

📌 Description:
{info['description']}

💊 Treatment:
{info['treatment']}

🛡 Prevention:
{info['prevention']}
"""

    else:
        response = f"""
🦠 रोग: {original_name}

📌 विवरण:
{info['description']}

💊 इलाज:
{info['treatment']}

🛡 रोकथाम:
{info['prevention']}
"""

    if user_input:
        response = f"🧑‍🌾 Farmer Input:\n{user_input}\n\n" + response

    return response