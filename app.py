from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
with open("model/Lgbm_Tuned_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("model/scaler.pkl", "rb") as scaler_file:
    scaler, selected_features = pickle.load(scaler_file)

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON request data
    data = request.json

    # Prepare input data
    features = [data.get(feature, 0) for feature in selected_features]
    features_scaled = scaler.transform([features])

    # Make prediction
    prediction = model.predict(features_scaled)
    return jsonify({'prediction': int(prediction[0])})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
