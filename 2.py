from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("rf_regression_model.pkl")

# Define the features the model expects
FEATURES = ['OBJECTID', 'rgn_id', 'are_km2', 'AO', 'BD', 'CP', 'CS', 'ECO', 'FIS', 'FP', 
            'HAB', 'ICO', 'Index_', 'LE', 'LIV', 'LSP', 'MAR', 'NP', 'SP', 'SPP', 'TR', 
            'trnd_sc', 'Shape__Area', 'Shape__Length']

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Ocean Health CW Predictor</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
        input { margin: 5px; padding: 8px; width: 200px; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
        .result { margin-top: 20px; padding: 15px; background: #e7f3ff; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Ocean Health Clean Waters Predictor</h1>
    <form action="/predict" method="post">
        <h3>Enter Ocean Health Indicators:</h3>
        {% for feature in features %}
        <input type="number" step="any" name="{{ feature }}" placeholder="{{ feature }}" required><br>
        {% endfor %}
        <button type="submit">Predict CW Score</button>
    </form>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form or JSON
        if request.is_json:
            data = request.json
        else:
            data = request.form.to_dict()
        
        # Convert to DataFrame
        input_data = pd.DataFrame([data])
        
        # Convert to numeric
        for col in FEATURES:
            input_data[col] = pd.to_numeric(input_data[col])
        
        # Make prediction
        prediction = model.predict(input_data[FEATURES])[0]
        
        return jsonify({
            'predicted_cw_score': float(prediction),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)