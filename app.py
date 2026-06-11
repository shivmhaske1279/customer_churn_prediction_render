import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

# Initialize the Flask application
app = Flask(__name__)

# Load the trained scikit-learn Naive Bayes model
MODEL_PATH = "naive_model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None
    print(f"Warning: {MODEL_PATH} not found. Please place it in the same directory.")

# Embedded HTML Template with Bootstrap for a clean UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Naive Bayes Model Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #f8f9fa; padding-top: 30px; }
        .card { border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .header-title { color: #2b3e50; font-weight: 700; margin-bottom: 25px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-8">
                <div class="card p-5 mb-5">
                    <h2 class="text-center header-title">📈 Customer Predictor Portal</h2>
                    <hr>
                    
                    {% if prediction is not none %}
                    <div class="alert alert-info text-center fs-4 fw-bold mb-4" role="alert">
                        Prediction Outcome: Class {{ prediction }}
                    </div>
                    {% endif %}

                    <form method="POST" action="/">
                        <div class="row g-3">
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Age</label>
                                <input type="number" step="any" class="form-control" name="age" value="30" required>
                            </div>
                            
                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Gender</label>
                                <select class="form-select" name="gender" required>
                                    <option value="0">Female (0)</option>
                                    <option value="1" selected>Male (1)</option>
                                </select>
                            </div>

                            <div class="col-md-6">
                                <label class="form-label fw-semibold">City (Encoded Number)</label>
                                <input type="number" class="form-control" name="city" value="0" required>
                            </div>

                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Tenure (Months)</label>
                                <input type="number" step="any" class="form-control" name="tenure_months" value="12" required>
                            </div>

                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Avg Order Value ($)</label>
                                <input type="number" step="any" class="form-control" name="avg_order_value" value="50.0" required>
                            </div>

                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Total Orders</label>
                                <input type="number" step="any" class="form-control" name="total_orders" value="5" required>
                            </div>

                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Last Purchase (Days Ago)</label>
                                <input type="number" step="any" class="form-control" name="last_purchase_days_ago" value="10" required>
                            </div>

                            <div class="col-md-6">
                                <label class="form-label fw-semibold">Support Tickets Raised</label>
                                <input type="number" step="any" class="form-control" name="support_tickets" value="0" required>
                            </div>

                            <div class="col-md-12">
                                <label class="form-label fw-semibold">Subscription Type (Encoded Number)</label>
                                <input type="number" class="form-control" name="subscription_type" value="0" required>
                            </div>
                        </div>

                        <button type="submit" class="btn btn-primary w-100 mt-4 py-2 fw-bold fs-5">Generate Prediction</button>
                    </form>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    
    if request.method == "POST":
        if model is None:
            return "Error: Model file is missing on the server.", 500
        
        try:
            # Extract and convert all 9 features from form input
            features = [
                float(request.form.get("age")),
                float(request.form.get("gender")),
                float(request.form.get("city")),
                float(request.form.get("tenure_months")),
                float(request.form.get("avg_order_value")),
                float(request.form.get("total_orders")),
                float(request.form.get("last_purchase_days_ago")),
                float(request.form.get("support_tickets")),
                float(request.form.get("subscription_type"))
            ]
            
            # Reshape features array for prediction: shape (1, 9)
            input_data = np.array([features])
            
            # Run inference
            pred_array = model.predict(input_data)
            prediction = int(pred_array[0])
            
        except Exception as e:
            prediction = f"Error processing input: {str(e)}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    # Runs the server locally on http://127.0.0.1:5000
    app.run(debug=True)
