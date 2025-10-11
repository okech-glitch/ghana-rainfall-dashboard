#!/usr/bin/env python3
"""
Minimal Rainfall Prediction API for Vercel
Ultra-simple Flask app with embedded HTML
"""

from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/')
def index():
    """Serve embedded HTML dashboard"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Rainfall Prediction Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 10px; margin: 20px 0; }
            input { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px; }
            button { background: #007bff; color: white; padding: 12px 20px; border: none; cursor: pointer; width: 100%; }
            #result { margin: 20px 0; padding: 15px; background: #e9ecef; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>🌧️ Rainfall Prediction Dashboard</h1>

        <div class="container">
            <h3>Weather Parameters</h3>
            <input type="number" id="temp" placeholder="Temperature (°C)" value="25">
            <input type="number" id="humidity" placeholder="Humidity (%)" value="65">
            <button onclick="predict()">Predict Rainfall</button>
        </div>

        <div id="result">Enter parameters and click predict</div>

        <p><small>LightGBM Ensemble Model | Demo Version</small></p>

        <script>
            async function predict() {
                const temp = document.getElementById('temp').value;
                const humidity = document.getElementById('humidity').value;

                document.getElementById('result').innerHTML = 'Loading...';

                try {
                    const response = await fetch('/api/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ temperature: temp, humidity: humidity })
                    });

                    const data = await response.json();
                    document.getElementById('result').innerHTML =
                        '<h3>Prediction: ' + data.prediction + '</h3>' +
                        '<p>Confidence: ' + Math.round(data.confidence * 100) + '%</p>';
                } catch (error) {
                    document.getElementById('result').innerHTML = 'Error: ' + error.message;
                }
            }
        </script>
    </body>
    </html>
    """

@app.route('/api/health')
def health():
    """Health check"""
    return jsonify({'status': 'ok', 'service': 'rainfall-prediction'})

@app.route('/api/predict', methods=['POST'])
def predict():
    """Simple prediction"""
    try:
        data = request.get_json() or {}

        temp = data.get('temperature', 25)
        humidity = data.get('humidity', 65)

        # Simple prediction logic
        if humidity > 70 or temp < 20:
            prediction = "HEAVYRAIN"
            confidence = 0.85
        elif humidity > 50:
            prediction = "MEDIUMRAIN"
            confidence = 0.65
        else:
            prediction = "NORAIN"
            confidence = 0.90

        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'model': 'demo'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
