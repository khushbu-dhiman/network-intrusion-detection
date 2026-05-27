from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ── Load model artifacts ──────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

try:
    model = joblib.load(os.path.join(BASE, "final_model.pkl"))
    scaler = joblib.load(os.path.join(BASE, "scaler.pkl"))
    selected_features = joblib.load(os.path.join(BASE, "selected_features.pkl"))
except Exception as e:
    print(f"Error loading with joblib: {e}")
    # Fallback to pickle with different encodings
    import pickle
    try:
        with open(os.path.join(BASE, "final_model.pkl"), "rb") as f:
            model = pickle.load(f, encoding='utf-8')
        with open(os.path.join(BASE, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f, encoding='utf-8')
        with open(os.path.join(BASE, "selected_features.pkl"), "rb") as f:
            selected_features = pickle.load(f, encoding='utf-8')
    except Exception as e2:
        raise Exception(f"Failed to load model artifacts: {e2}")

# ── Feature definitions (same order as training) ─────────────────────────────
ALL_FEATURES = [
    "protocol", "flow_duration", "total_fwd_packets",
    "fwd_packets_length_total", "bwd_packets_length_total",
    "fwd_packet_length_max", "fwd_packet_length_mean",
    "bwd_packet_length_max", "bwd_packet_length_min",
    "flow_bytes/s", "flow_packets/s",
    "flow_iat_mean", "flow_iat_std", "flow_iat_min",
    "fwd_iat_min", "bwd_iat_total", "bwd_iat_mean",
    "bwd_iat_std", "bwd_iat_max", "bwd_iat_min",
    "bwd_packets/s", "fin_flag_count", "down/up_ratio",
    "init_fwd_win_bytes", "init_bwd_win_bytes",
    "fwd_act_data_packets", "fwd_seg_size_min",
    "active_mean", "active_max", "active_min",
]

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", features=ALL_FEATURES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Build full feature vector
        values = []
        for feat in ALL_FEATURES:
            val = data.get(feat)
            if val is None:
                return jsonify({"error": f"Missing feature: {feat}"}), 400
            values.append(float(val))

        # Create DataFrame with all features
        full_df = pd.DataFrame([values], columns=ALL_FEATURES)
        
        # Select only the features used during training
        selected_df = full_df[selected_features]

        # Scale using the fitted scaler (which expects feature names)
        scaled = scaler.transform(selected_df)

        # Predict
        pred = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0] if hasattr(model, "predict_proba") else None

        result = {"prediction": pred}
        
        if proba is not None:
            classes = model.classes_
            result["probabilities"] = {}
            for c, p in zip(classes, proba):
                result["probabilities"][str(c)] = round(float(p) * 100, 2)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)

    
