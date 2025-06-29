# src/predict.py
import mlflow.pyfunc
from flask import Flask, request, jsonify
import pandas as pd
import os
import pickle
from mlflow.tracking import MlflowClient
import mlflow.artifacts


mlflow.set_tracking_uri("http://localhost:5000")

app = Flask(__name__)
model = None

def load_model():
    """Loads the MLflow model."""
    global model
    try:  
        print(f"Loading model from")
        from joblib import load
        model = load("model.pkl")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None # Ensure model is None if loading fails

@app.route('/predict', methods=['POST'])
def predict():
    load_model()
    if model is None:
        return jsonify({"error": "Model not loaded. Server might be initializing or encountered an error."}), 500

    try:
        json_ = request.json
        # Assuming input is a list of lists or dicts matching feature names
        # For Iris, it expects a list of 4 float values, e.g., [[5.1, 3.5, 1.4, 0.2]]
        feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

        if isinstance(json_, list):
            if not json_ or not isinstance(json_[0], list) or len(json_[0]) != len(feature_names):
                return jsonify({"error": f"Invalid input format for list. Expected list of lists with {len(feature_names)} features, e.g., [[5.1, 3.5, 1.4, 0.2]]"}), 400
            input_df = pd.DataFrame(json_, columns=feature_names)
        elif isinstance(json_, dict):
            if not all(col in json_ for col in feature_names):
                return jsonify({"error": f"Invalid input format for dict. Expected dictionary with keys: {feature_names}"}), 400
            input_df = pd.DataFrame([json_])
        else:
            return jsonify({"error": "Invalid input format. Expected list of lists or dictionary."}), 400

        predictions = model.predict(input_df)
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    load_model()
    # For local testing, use a development server directly:
    # app.run(host='0.0.0.0', port=5000, debug=True)
    # For production in Docker, Gunicorn will be used,
    # so direct `app.run` is typically not needed here if Dockerfile CMD uses Gunicorn.
    # However, for direct Python execution testing:
    # gunicorn command will handle starting the app correctly in Docker.
    # If you want to test this predict.py script directly without Docker:
    # pip install gunicorn flask
    # gunicorn --bind 0.0.0.0:5000 predict:app
    # app.run(host='0.0.0.0', port=5000) # This line is often left out if Gunicorn is used by default in Docker.