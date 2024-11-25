from flask import Flask, request, jsonify
import lightgbm as lgb
import pandas as pd
import json
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load LightGBM model
model = lgb.Booster(model_file='data/lgb_model.txt')

# Load unique values JSON
with open('data/unique_values.json', 'r') as f:
    unique_values = json.load(f)


@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "LightGBM Model API is running!"})

@app.route('/unique_values', methods=['GET'])
def get_unique_values():
    return jsonify(unique_values)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        input_data = request.get_json()

        # Ensure input data exists
        if "data" not in input_data:
            return jsonify({"error": "Input data is missing"}), 400

        # Convert input to NumPy array
        input_array = np.array(input_data["data"])

        # Reshape to 2D array if input is 1D
        if input_array.ndim == 1:
            input_array = input_array.reshape(1, -1)

        # Validate input shape matches model expectations
        expected_features = len(model.feature_name())
        if input_array.shape[1] != expected_features:
            return jsonify({"error": f"Expected {expected_features} features, but got {input_array.shape[1]}"}), 400

        # Make predictions
        prediction = model.predict(input_array)

        # For classification, return the class with the highest probability
        if model.params.get("objective", "").startswith("multiclass"):
            predicted_class = int(np.argmax(prediction, axis=1)[0])
            return jsonify({"prediction": predicted_class})
        
        # For regression, return the raw prediction
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 400




if __name__ == '__main__':
    app.run(debug=True)

