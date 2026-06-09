import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

# Load the model safely
MODEL_PATH = "naive_model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None

# Modern, attractive HTML layout with shadows and clean components
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Prediction Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #f4f6f9;
            --card-bg: #ffffff;
            --text-main: #2d3748;
            --text-muted: #718096;
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --shadow-sm: 0 2px 4px rgba(0,0,0,0.05);
            --shadow-md: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
            --shadow-lg: 0 20px 25px -5px rgba(0,0,0,0.1), 0 10px 10px -5px rgba(0,0,0,0.04);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.6;
            padding: 40px 20px;
        }

        .container {
            max-width: 850px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
        }

        .header h1 {
            font-size: 2.25rem;
            color: #1a202c;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .header p {
            color: var(--text-muted);
            font-size: 1.1rem;
        }

        .card {
            background: var(--card-bg);
            border-radius: 16px;
            box-shadow: var(--shadow-lg);
            padding: 40px;
            margin-bottom: 30px;
            border: 1px solid rgba(226, 232, 240, 0.8);
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 24px;
        }

        @media (max-width: 600px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .form-group label {
            font-size: 0.9rem;
            font-weight: 600;
            color: #4a5568;
            margin-bottom: 8px;
            text-transform: capitalize;
        }

        .form-group input, .form-group select {
            padding: 12px 16px;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            font-size: 1rem;
            font-family: inherit;
            color: var(--text-main);
            background-color: #fff;
            transition: all 0.2s ease;
            box-shadow: var(--shadow-sm);
        }

        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.15);
        }

        .btn-submit {
            grid-column: span 2;
            background: var(--primary);
            color: white;
            border: none;
            padding: 14px;
            border-radius: 8px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s ease;
            box-shadow: var(--shadow-md);
            margin-top: 10px;
        }

        @media (max-width: 600px) {
            .btn-submit {
                grid-column: span 1;
            }
        }

        .btn-submit:hover {
            background: var(--primary-hover);
        }

        .result-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 16px;
            padding: 30px;
            text-align: center;
            box-shadow: var(--shadow-lg);
            margin-top: 30px;
            animation: fadeIn 0.4s ease-out;
        }

        .result-card h2 {
            font-size: 1.8rem;
            margin-bottom: 10px;
            font-weight: 700;
        }

        .result-card p {
            font-size: 1.2rem;
            opacity: 0.9;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Predictive Analytics Dashboard</h1>
        <p>Enter the features below to evaluate the Naive Bayes model output</p>
    </div>

    <div class="card">
        <form method="POST" action="/predict">
            <div class="grid">
                <div class="form-group">
                    <label for="age">Age</label>
                    <input type="number" id="age" name="age" required min="0" placeholder="e.g. 34">
                </div>
                <div class="form-group">
                    <label for="gender">Gender (Encoded)</label>
                    <input type="number" id="gender" name="gender" required placeholder="e.g. 0 or 1" step="any">
                </div>
                
                <div class="form-group">
                    <label for="city">City (Encoded)</label>
                    <input type="number" id="city" name="city" required placeholder="e.g. 1, 2, 3" step="any">
                </div>
                <div class="form-group">
                    <label for="tenure_months">Tenure Months</label>
                    <input type="number" id="tenure_months" name="tenure_months" required min="0" placeholder="e.g. 12">
                </div>
                
                <div class="form-group">
                    <label for="avg_order_value">Avg Order Value ($)</label>
                    <input type="number" id="avg_order_value" name="avg_order_value" required step="any" placeholder="e.g. 85.50">
                </div>
                <div class="form-group">
                    <label for="total_orders">Total Orders</label>
                    <input type="number" id="total_orders" name="total_orders" required min="0" placeholder="e.g. 5">
                </div>
                
                <div class="form-group">
                    <label for="last_purchase_days_ago">Last Purchase (Days Ago)</label>
                    <input type="number" id="last_purchase_days_ago" name="last_purchase_days_ago" required min="0" placeholder="e.g. 14">
                </div>
                <div class="form-group">
                    <label for="support_tickets">Support Tickets</label>
                    <input type="number" id="support_tickets" name="support_tickets" required min="0" placeholder="e.g. 0">
                </div>
                
                <div class="form-group" style="grid-column: span 2;">
                    <label for="subscription_type">Subscription Type (Encoded)</label>
                    <input type="number" id="subscription_type" name="subscription_type" required placeholder="e.g. 0, 1, 2" step="any">
                </div>

                <button type="submit" class="btn-submit">Run Model Prediction</button>
            </div>
        </form>
    </div>

    {% if prediction is not none %}
    <div class="result-card">
        <h2>Prediction Result</h2>
        <p>The model predicted class: <strong>{{ prediction }}</strong></p>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return "Model not loaded. Please ensure naive_model.pkl is in the root folder.", 500
    
    try:
        # Extract inputs in the exact sequence expected by your .pkl file
        features = [
            float(request.form.get('age')),
            float(request.form.get('gender')),
            float(request.form.get('city')),
            float(request.form.get('tenure_months')),
            float(request.form.get('avg_order_value')),
            float(request.form.get('total_orders')),
            float(request.form.get('last_purchase_days_ago')),
            float(request.form.get('support_tickets')),
            float(request.form.get('subscription_type'))
        ]
        
        # Format for Scikit-Learn prediction (2D array)
        final_features = [np.array(features)]
        prediction = model.predict(final_features)[0]
        
        return render_template_string(HTML_TEMPLATE, prediction=int(prediction))

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
