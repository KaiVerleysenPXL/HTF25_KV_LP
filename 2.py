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
    <title>Ocean Health Predictor</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: 50px auto; padding: 20px; }
        h1 { text-align: center; color: #007bff; }
        .upload-section { background: #f8f9fa; padding: 30px; border-radius: 10px; margin-bottom: 30px; }
        input[type="file"] { margin: 20px 0; padding: 10px; }
        .feature-selector { margin: 20px 0; }
        .feature-selector h3 { margin-bottom: 15px; }
        .radio-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
        .radio-grid label { padding: 10px; border: 2px solid #ddd; border-radius: 5px; cursor: pointer; display: block; }
        .radio-grid label:hover { background: #e9ecef; }
        .radio-grid input[type="radio"] { margin-right: 8px; }
        button { padding: 15px 40px; background: #007bff; color: white; border: none; cursor: pointer; font-size: 16px; border-radius: 5px; }
        button:hover { background: #0056b3; }
        .center { text-align: center; }
    </style>
    <script>
        function showFeatureSelector() {
            const fileInput = document.querySelector('input[type="file"]');
            const selector = document.getElementById('featureSelector');
            if (fileInput.files.length > 0) {
                selector.style.display = 'block';
            }
        }
    </script>
</head>
<body>
    <h1>🌊 Ocean Health Index Predictor</h1>
    <div class="upload-section">
        <form action="/predict_csv" method="post" enctype="multipart/form-data">
            <div class="center">
                <label for="fileInput"><strong>Upload CSV File:</strong></label><br>
                <input type="file" id="fileInput" name="file" accept=".csv" required onchange="showFeatureSelector()"><br>
            </div>
            
            <div id="featureSelector" class="feature-selector" style="display: none;">
                <h3>Select Feature to Display on Map:</h3>
                <div class="radio-grid">
                    <label>
                        <input type="radio" name="feature" value="Index_" checked>
                        Overall Health (Index)
                    </label>
                    <label>
                        <input type="radio" name="feature" value="CW">
                        Clean Waters (CW)
                    </label>
                    <label>
                        <input type="radio" name="feature" value="BD">
                        Biodiversity (BD)
                    </label>
                    <label>
                        <input type="radio" name="feature" value="FIS">
                        Fisheries (FIS)
                    </label>
                    <label>
                        <input type="radio" name="feature" value="AO">
                        Artisanal Fishing (AO)
                    </label>
                    <label>
                        <input type="radio" name="feature" value="HAB">
                        Habitat (HAB)
                    </label>
                    <label>
                        <input type="radio" name="feature" value="SPP">
                        Species (SPP)
                    </label>
                    <label>
                        <input type="radio" name="feature" value="ECO">
                        Economies (ECO)
                    </label>
                    <label>
                        <input type="radio" name="feature" value="TR">
                        Tourism & Recreation (TR)
                    </label>
                    <label>
                        <input type="radio" name="feature" value="trnd_sc">
                        Trend Score
                    </label>
                    <label>
                        <input type="radio" name="feature" value="CP">
                        Coastal Protection (CP)
                    </label>
                    <label>
                        <input type="radio" name="feature" value="CS">
                        Carbon Storage (CS)
                    </label>
                </div>
            </div>
            
            <div class="center" style="margin-top: 20px;">
                <button type="submit">Analyze & Visualize</button>
            </div>
        </form>
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

def create_map_html(results, feature_name):
    """Create the map visualization HTML"""
    # Convert results to JSON for JavaScript
    import json
    results_json = json.dumps(results)
    
    feature_labels = {
        'Index_': 'Overall Ocean Health',
        'CW': 'Clean Waters',
        'BD': 'Biodiversity',
        'FIS': 'Fisheries',
        'AO': 'Artisanal Fishing Opportunities',
        'HAB': 'Habitat',
        'SPP': 'Species',
        'ECO': 'Coastal Livelihoods & Economies',
        'TR': 'Tourism & Recreation',
        'trnd_sc': 'Trend Score',
        'CP': 'Coastal Protection',
        'CS': 'Carbon Storage'
    }
    
    label = feature_labels.get(feature_name, feature_name)
    
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{label} - World Map</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; font-family: Arial; }}
        #map {{ height: 100vh; width: 100%; }}
        .legend {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            z-index: 1000;
            max-height: 80vh;
            overflow-y: auto;
            min-width: 250px;
        }}
        .legend h3 {{ margin-top: 0; color: #007bff; }}
        .legend-item {{
            margin: 8px 0;
            padding: 8px;
            border-left: 4px solid;
            background: #f9f9f9;
        }}
        .gradient-bar {{
            height: 20px;
            background: linear-gradient(to right, #ff0000, #ffff00, #00ff00);
            margin: 10px 0;
            border-radius: 3px;
        }}
        .gradient-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 12px;
        }}
        .back-btn {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: #007bff;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            text-decoration: none;
            z-index: 1000;
        }}
        .back-btn:hover {{ background: #0056b3; }}
    </style>
</head>
<body>
    <a href="/" class="back-btn">← Upload New Data</a>
    <div id="map"></div>
    <div class="legend">
        <h3>{label}</h3>
        <div class="gradient-bar"></div>
        <div class="gradient-labels">
            <span>0 (Low)</span>
            <span>50</span>
            <span>100 (High)</span>
        </div>
        <div id="legend-items" style="margin-top: 20px;"></div>
    </div>
    
    <script>
        const predictions = {results_json};
        
        const map = L.map('map').setView([20, 0], 2);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);

        function getColor(score) {{
            if (score >= 75) return '#00ff00';
            if (score >= 50) return '#7fff00';
            if (score >= 25) return '#ffff00';
            return '#ff0000';
        }}

        const predictionMap = {{}};
        predictions.forEach(pred => {{
            if (pred.rgn_key) {{
                predictionMap[pred.rgn_key] = pred;
            }}
        }});

        fetch('https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json')
            .then(response => response.json())
            .then(data => {{
                function style(feature) {{
                    const countryCode = feature.id;
                    const pred = predictionMap[countryCode];
                    
                    return {{
                        fillColor: pred ? getColor(pred.value) : '#cccccc',
                        weight: 1,
                        opacity: 1,
                        color: 'white',
                        fillOpacity: pred ? 0.7 : 0.3
                    }};
                }}

                function onEachFeature(feature, layer) {{
                    const countryCode = feature.id;
                    const pred = predictionMap[countryCode];
                    
                    if (pred) {{
                        layer.bindPopup(`
                            <b>${{pred.rgn_nam || pred.rgn_key || 'Region ' + pred.rgn_id}}</b><br>
                            {label}: ${{pred.value ? pred.value.toFixed(2) : 'N/A'}}<br>
                            ${{pred.trnd_sc !== null ? 'Trend: ' + pred.trnd_sc.toFixed(2) : ''}}
                        `);
                        
                        layer.on({{
                            mouseover: function(e) {{
                                const layer = e.target;
                                layer.setStyle({{
                                    weight: 3,
                                    color: '#666',
                                    fillOpacity: 0.9
                                }});
                            }},
                            mouseout: function(e) {{
                                geojsonLayer.resetStyle(e.target);
                            }}
                        }});
                    }}
                }}

                const geojsonLayer = L.geoJSON(data, {{
                    style: style,
                    onEachFeature: onEachFeature
                }}).addTo(map);
            }})
            .catch(error => {{
                console.error('Error loading country boundaries:', error);
            }});

        const legendItems = document.getElementById('legend-items');
        
        predictions.forEach(pred => {{
            const color = getColor(pred.value);
            
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.style.borderLeftColor = color;
            item.innerHTML = `
                <strong>${{pred.rgn_nam || pred.rgn_key || 'Region ' + pred.rgn_id}}</strong><br>
                Score: ${{pred.value ? pred.value.toFixed(2) : 'N/A'}}
            `;
            legendItems.appendChild(item);
        }});
    </script>
</body>
</html>
"""

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded', 'status': 'failed'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'status': 'failed'}), 400
        
        # Get selected feature to display
        selected_feature = request.form.get('feature', 'Index_')
        
        # Read CSV
        csv_data = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))
        
        # Get the actual values for the selected feature
        feature_values = df[selected_feature].tolist() if selected_feature in df.columns else [None] * len(df)
        
        # Store region info
        rgn_names = df['rgn_nam'].tolist() if 'rgn_nam' in df.columns else [None] * len(df)
        rgn_ids = df['rgn_id'].tolist() if 'rgn_id' in df.columns else list(range(len(df)))
        rgn_keys = df['rgn_key'].tolist() if 'rgn_key' in df.columns else [None] * len(df)
        trend_scores = df['trnd_sc'].tolist() if 'trnd_sc' in df.columns else [None] * len(df)
        
        # Format results with actual values from CSV
        results = []
        for i in range(len(df)):
            results.append({
                'rgn_id': int(rgn_ids[i]) if rgn_ids[i] is not None else i,
                'rgn_nam': rgn_names[i],
                'rgn_key': rgn_keys[i],
                'value': float(feature_values[i]) if feature_values[i] is not None else None,
                'trnd_sc': float(trend_scores[i]) if trend_scores[i] and trend_scores[i] is not None else None
            })
        
        # Create visualization HTML
        return render_template_string(create_map_html(results, selected_feature))
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)