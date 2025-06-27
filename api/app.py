# api/app.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# api/app.py

from dotenv import load_dotenv
load_dotenv()

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify, abort
import joblib
import numpy as np

# Hard-coded API key (as requested)
API_KEY = "S3cReT_Ph1ShK3y"

# Paths (absolute)
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

# Import feature extractor
from utils.feature_utils import lexical_features

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    """
    Protected endpoint requiring x-api-key header.
    Expects JSON body: {"url": "http://..."}.
    Returns JSON: {"url", "prediction", "confidence"}.
    """
    # API key header check (case-insensitive)
    key = request.headers.get('x-api-key') or request.headers.get('X-API-KEY')
    if key != API_KEY:
        app.logger.warning(f"Unauthorized access attempt with key: {key}")
        abort(401, description='Unauthorized: invalid API key')

    # Parse request JSON
    data = request.get_json(force=True)
    url = data.get('url', '').strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Feature extraction
    feats = lexical_features(url)
    X = np.array([list(feats.values())], dtype=float)

    # Load scaler and model
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)

    # Predict
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    proba = float(model.predict_proba(X_scaled)[0,1])

    label = 'phishing' if pred == 1 else 'legitimate'
    return jsonify({
        "url": url,
        "prediction": label,
        "confidence": proba
    }), 200

if __name__ == '__main__':
    # Local development server (not for production)
    app.run(host='0.0.0.0', port=5000, debug=True)