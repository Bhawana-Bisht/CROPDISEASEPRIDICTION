import os
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="""
You are an agriculture expert.

Disease: Bacterial Blight in rice
Give:
- Simple explanation
- Symptoms
- Treatment
- Prevention
"""
)

print(response.text)