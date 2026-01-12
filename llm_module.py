import json

def generate_llm_response(disease_name):
    with open("disease_info.json", "r") as f:
        data = json.load(f)

    info = data[disease_name]

    response = f"""
🦠 Disease Detected: {disease_name}

📌 Description:
{info['description']}

💊 Treatment:
{info['treatment']}

🛡 Prevention:
{info['prevention']}
"""
    return response
