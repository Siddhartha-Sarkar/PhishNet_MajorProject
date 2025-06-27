import os
import sys
import joblib
import numpy as np
from flask import Flask, request, jsonify, abort
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your custom function
from utils.feature_utils import lexical_features

# Constants
API_KEY      = os.getenv("PHISH_DETECTOR_API_KEY", "S3cReT_Ph1ShK3y")
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH   = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
SCALER_PATH  = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Welcome to PhishNet API. Use the /predict endpoint."
    }), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    # Check API key
    key = request.headers.get('x-api-key') or request.headers.get('X-API-KEY')
    if key != API_KEY:
        app.logger.warning(f"Unauthorized access attempt with key: {key}")
        abort(401, description='Unauthorized: Invalid API key')

    # Validate input JSON
    data = request.get_json(force=True)
    url = data.get('url', '').strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        # Extract features
        feats = lexical_features(url)
        X = np.array([list(feats.values())], dtype=float)

        # Load scaler and model
        scaler = joblib.load(SCALER_PATH)
        model  = joblib.load(MODEL_PATH)

        # Predict
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        proba = float(model.predict_proba(X_scaled)[0, 1])

        label = 'phishing' if pred == 1 else 'legitimate'

        return jsonify({
            "url": url,
            "prediction": label,
            "confidence": round(proba, 4)
        }), 200

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
