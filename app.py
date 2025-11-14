from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import pandas as pd

app = Flask(__name__)

# Load model, scaler and feature order
MODEL_PATH = os.path.join("model", "heart_disease_model.pkl")
SCALER_PATH = os.path.join("model", "scaler.joblib")
FEATURE_ORDER_PATH = os.path.join("model", "feature_order.txt")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_order = list(pd.read_csv(FEATURE_ORDER_PATH, header=None)[0])

@app.route("/")
def home():
    return render_template("index.html", output=None)

@app.route("/predict", methods=["POST"])
def predict():
    # Extract inputs from form — change names if your form differs
    # Example expecting these exact 7 features: cp, thalach, oldpeak, exang, ca, thal, slope
    try:
        cp = float(request.form["cp"])
        thalach = float(request.form["thalach"])
        oldpeak = float(request.form["oldpeak"])
        exang = float(request.form["exang"])
        ca = float(request.form["ca"])
        thal = float(request.form["thal"])
        slope = float(request.form["slope"])
    except Exception as e:
        return render_template("index.html", output=f"Invalid input: {e}")

    # Build array in the correct order used during training.
    # If full-feature model was trained, ensure you supply all features in the same order.
    # Here we assume these 7 features model or that the model expects this order:
    X = np.array([[cp, thalach, oldpeak, exang, ca, thal, slope]])
    # If you trained on full set, you must construct array of full length in feature_order order.

    # Scale features if scaler exists and model was trained on scaled features (for LR/KNN)
    # If your final_model is RandomForest trained on raw features, do NOT scale.
    # Here we assume model expects raw features (RandomForest). If you used scaler, uncomment:
    # X = scaler.transform(X)

    pred = model.predict(X)[0]
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0][1]

    result = "Heart Disease Detected" if pred == 1 else "No Heart Disease"
    if prob is not None:
        result = f"{result} (probability: {prob:.2f})"
    return render_template("index.html", output=result)

if __name__ == "__main__":
    app.run(debug=True)
