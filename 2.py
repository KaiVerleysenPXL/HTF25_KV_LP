from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import io

app = Flask(__name__)

# Load the trained model
model = joblib.load("rf_regression_model.pkl")

# Define the features the model expects (excluding CW)
FEATURES = ['OBJECTID', 'rgn_id', 'are_km2', 'AO', 'BD', 'CP', 'CS', 'ECO', 'FIS', 'FP', 
            'HAB', 'ICO', 'Index_', 'LE', 'LIV', 'LSP', 'MAR', 'NP', 'SP', 'SPP', 'TR', 
            'trnd_sc', 'Shape__Area', 'Shape__Length']

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Ocean Health CW Predictor</title>
    <style>
        body { font-family: Arial; max-width: 600px; margin: 100px auto; padding: 20px; text-align: center; }
        input[type="file"] { margin: 20px; padding: 10px; }
        button { padding: 15px 30px; background: #007bff; color: white; border: none; cursor: pointer; font-size: 16px; }
        button:hover { background: #0056b3; }
    </style>
</head>
<body>
    <h1>Upload CSV for CW Predictions</h1>
    <form action="/predict_csv" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".csv" required><br>
        <button type="submit">Upload and Predict</button>
    </form>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded', 'status': 'failed'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'status': 'failed'}), 400
        
        # Read CSV
        csv_data = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))
        
        # Store actual CW values if present
        has_actual_cw = 'CW' in df.columns
        actual_cw_values = df['CW'].tolist() if has_actual_cw else [None] * len(df)
        
        # Store region info
        rgn_names = df['rgn_nam'].tolist() if 'rgn_nam' in df.columns else [None] * len(df)
        rgn_ids = df['rgn_id'].tolist() if 'rgn_id' in df.columns else list(range(len(df)))
        
        # Prepare features for prediction
        numeric_cols = df.select_dtypes(include=['number']).columns
        available_features = [f for f in FEATURES if f in numeric_cols]
        
        if len(available_features) == 0:
            return jsonify({'error': 'No valid features found in CSV', 'status': 'failed'}), 400
        
        X = df[available_features]
        
        # Make predictions
        predictions = model.predict(X)
        
        # Format results
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                'rgn_id': int(rgn_ids[i]) if rgn_ids[i] is not None else i,
                'rgn_nam': rgn_names[i],
                'predicted_cw': float(pred),
                'actual_cw': float(actual_cw_values[i]) if actual_cw_values[i] is not None else None
            })
        
        return jsonify({
            'status': 'success',
            'predictions': results,
            'total_regions': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)