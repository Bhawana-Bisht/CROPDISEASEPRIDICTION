from flask import Flask, request, jsonify
import os
import sys

# ---------------------------------
# Fix Paths (VERY IMPORTANT)
# ---------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# 🔥 Correct path add (src folder)
SRC_PATH = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_PATH)

# Import using package structure
from predict.predict_disease import predict_disease

app = Flask(__name__)

# ---------------------------------
# Temp Upload Folder
# ---------------------------------
UPLOAD_FOLDER = os.path.join(BASE_DIR, "temp_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------------------------
# Home Route
# ---------------------------------
@app.route("/")
def home():
    return "🌾 Rice Disease Detection API Running..."


# ---------------------------------
# Predict Route
# ---------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    # -----------------------------
    # Check File
    # -----------------------------
    if "file" not in request.files:
        return jsonify({
            "error": "No file uploaded"
        })

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "Empty file name"
        })

    # -----------------------------
    # Farmer Input
    # -----------------------------
    user_query = request.form.get("query", "").strip()

    # -----------------------------
    # Language Input
    # -----------------------------
    language = request.form.get("language", "Hindi")

    # -----------------------------
    # Validation
    # -----------------------------
    if user_query == "":
        return jsonify({
            "error": "Please enter farmer problem"
        })

    if len(user_query) < 10:
        return jsonify({
            "error": "Problem should be at least 10 characters"
        })

    # -----------------------------
    # Save Temporary Image
    # -----------------------------
    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(file_path)

    try:

        # -----------------------------
        # Predict Disease
        # -----------------------------
        result = predict_disease(
            file_path,
            user_query,
            language
        )

        # -----------------------------
        # Return Result
        # -----------------------------
        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        })

    finally:

        # -----------------------------
        # Delete Temp File
        # -----------------------------
        if os.path.exists(file_path):
            os.remove(file_path)


# ---------------------------------
# Run App
# ---------------------------------
if __name__ == "__main__":
    app.run(debug=True)