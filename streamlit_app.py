import streamlit as st
import requests
from PIL import Image
import os
import time
from google import genai
from reportlab.pdfgen import canvas
from src.predict.llm_module import generate_llm_response

# -----------------------------
# Gemini Setup
# -----------------------------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# -----------------------------
# DARK MODE STATE
# -----------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Rice Disease Detection AI",
    page_icon="🌾",
    layout="wide"
)

# -----------------------------
# DARK MODE TOGGLE
# -----------------------------
dark = st.sidebar.toggle("🌙 Dark Mode")
st.session_state.dark_mode = dark

if st.session_state.dark_mode:
    bg = "#0f172a"
    card = "#1e293b"
    text = "white"
else:
    bg = "#f8fafc"
    card = "white"
    text = "#111827"

# -----------------------------
# GLOBAL CSS
# -----------------------------
st.markdown(f"""
<style>
.stApp {{
    background:{bg};
    color:{text};
}}

div[data-testid="stSidebar"] {{
    background: #14532d;
    color: white;
}}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("🌾 Farmer Dashboard")
    st.markdown("---")
    st.markdown("""
### 📌 Features

✅ Rice Disease Detection  
✅ AI Farmer Advice  
✅ Multi-language Support  
✅ Smart Crop Analysis  
✅ Confidence Prediction  
✅ PDF Report  
""")
    st.success("✅ AI System Active")

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div style="
text-align:center;
padding:25px;
background:linear-gradient(135deg,#1b5e20,#43a047);
border-radius:20px;
color:white;
box-shadow:0px 10px 25px rgba(0,0,0,0.2);
">
<h1>🌾 Rice Disease Detection AI</h1>
<p>Smart Agriculture Assistant powered by AI + Deep Learning</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# LANGUAGE
# -----------------------------
language = st.selectbox(
    "🌐 Language Select Karein",
    ["🇮🇳 Hindi", "🇬🇧 English", "🗣 Hinglish"]
)

selected_language = language.split(" ")[1]

# -----------------------------
# LAYOUT
# -----------------------------
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown(f"""
    <div style="background:{card};padding:20px;border-radius:20px;">
    <h3>📸 Upload Rice Leaf Image</h3>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)

with right_col:
    st.markdown(f"""
    <div style="background:{card};padding:20px;border-radius:20px;">
    <h3>🧑‍🌾 Farmer Input</h3>
    """, unsafe_allow_html=True)

    user_query = st.text_area("Apni problem likhein", height=180)

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# GEMINI FUNCTION
# -----------------------------
def get_llm_advice(disease, user_input, language):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"""
You are an agriculture expert.

Disease: {disease}
Language: {language}
Farmer Input: {user_input}

Give:
- Simple explanation
- Symptoms
- Treatment
- Prevention

Make it farmer friendly.
"""
        )
        return response.text
    except Exception as e:
        st.warning("Internet unavailable. Using local knowledge base.")
        return generate_llm_response(
            disease,
            user_input,
            language
        )

# -----------------------------
# PDF GENERATOR
# -----------------------------
def create_pdf(disease, confidence, advice, user_input):
    file_name = "report.pdf"
    c = canvas.Canvas(file_name)

    c.drawString(100, 800, "Rice Disease Detection Report")
    c.drawString(100, 770, f"Disease: {disease}")
    c.drawString(100, 750, f"Confidence: {confidence}%")
    c.drawString(100, 730, f"User Input: {user_input}")
    c.drawString(100, 690, "AI Advice:")
    c.drawString(100, 670, advice[:800])

    c.save()
    return file_name

# -----------------------------
# BUTTON
# -----------------------------
predict_btn = st.button("🔍 Predict Disease", use_container_width=True)

# -----------------------------
# PROCESSING
# -----------------------------
if predict_btn:

    if uploaded_file is None:
        st.warning("⚠ Please upload image")

    elif user_query.strip() == "":
        st.warning("⚠ Enter problem")

    else:

        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }

        data = {
            "query": user_query,
            "language": selected_language
        }

        # ---------------- LOADING ANIMATION ----------------
        with st.status("🤖 AI analyzing your crop...", expanded=True) as status:
            st.write("Processing image...")
            time.sleep(1)
            st.write("Detecting disease...")
            time.sleep(1)
            st.write("Generating AI advice...")
            time.sleep(1)

        try:
            response = requests.post(
                "http://127.0.0.1:5000/predict",
                files=files,
                data=data
            )

            result = response.json()

            disease = result.get("disease", "N/A")
            confidence = result.get("confidence", 0)

            formatted_disease = disease.replace("_", " ").title()

            # ---------------- LLM ----------------
            advice = get_llm_advice(formatted_disease, user_query, selected_language)

            # ---------------- RESULT ----------------
            st.markdown(f"""
            <div style="background:{card};padding:25px;border-radius:20px;">
            <h2>🌾 Prediction Result</h2>
            <hr>
            <h3>Disease: {formatted_disease}</h3>
            <h3>Confidence: {confidence:.2f}%</h3>
            <h3>Language: {selected_language}</h3>
            </div>
            """, unsafe_allow_html=True)

            st.progress(float(confidence)/100)

            # ---------------- METRICS ----------------
            m1, m2, m3 = st.columns(3)
            m1.metric("🌾 Disease", formatted_disease)
            m2.metric("📊 Confidence", f"{confidence:.2f}%")
            m3.metric("🌐 Language", selected_language)

            # ---------------- CHATGPT STYLE BOX ----------------
            st.markdown("## 🤖 Farmer Advice")

            st.markdown(f"""
            <div style="
            background:{card};
            padding:20px;
            border-left:6px solid #22c55e;
            border-radius:12px;
            line-height:1.6;
            ">
            💬 {advice}
            </div>
            """, unsafe_allow_html=True)

            # ---------------- PDF ----------------
            pdf_file = create_pdf(formatted_disease, confidence, advice, user_query)

            st.download_button(
                "📄 Download PDF Report",
                open(pdf_file, "rb"),
                file_name="rice_report.pdf"
            )

            # ---------------- ARCHITECTURE ----------------
            st.markdown("## 🧠 System Architecture")

            st.code("""
User → Streamlit UI
     ↓
Flask API
     ↓
CNN Model
     ↓
Gemini LLM
     ↓
Result Display
""")

        except Exception as e:
            st.error(f"Error: {e}")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#64748b;">
🌱 Smart Agriculture AI Project | Final Year MCA Project
</div>
""", unsafe_allow_html=True)