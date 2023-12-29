"""
Flask app for Lending Club data prediction.
"""
import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from sklearn import set_config

set_config(transform_output="pandas")


# Load the pre-processing pipelines with the models from disk
model_version__basic_info = "1.0"
pipeline_basic_info = joblib.load(
    "./models/classifier-1--without_credit_history.pickle"
)

model_version__with_history = "1.0"
pipeline_with_history = joblib.load(
    "./models/classifier-2--with_credit_history.pickle"
)

app = Flask(__name__)


# Main page
@app.route("/", methods=["GET"])
def hello():
    """Create the main page for app."""

    # Get Cloud Run environment variables.
    service = os.environ.get("K_SERVICE", "Unknown service")
    revision = os.environ.get("K_REVISION", "Unknown revision")

    return render_template(
        "index.html",
        Service=service,
        Revision=revision,
        model_1_version=model_version__basic_info,
        model_2_version=model_version__with_history,
    )


# To test if the server is up and running
@app.route("/test", methods=["GET"])
def test_server():
    """Test page without html template."""
    return "OK - Server is up and running!"


# To get the model's predictions
@app.route("/api/predict", methods=["POST"])
def predict_loan_status():
    """API endpoint to make predictions based on info from application (basic)."""
    try:
        # Get JSON data from the request and convert it to a data frame
        # NOTE: Data validation step is not implemented.
        df = pd.DataFrame.from_dict(request.json)

        # Perform predictions using the pipeline (model)
        prediction = pipeline_basic_info.predict(df)

        # Probability to default
        probability = pipeline_basic_info.predict_proba(df)[:, 1]

        # Return the results as JSON
        return jsonify({
            "model": "Credit default prediction (no credit history is needed)",
            "model_version": model_version__basic_info,
            "model_output_options": "1: financial difficulties, 0: no financial difficulties",
            "prediction": prediction.tolist(),
            "probability": probability.tolist(),
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/predict-with-credit-history", methods=["POST"])
def predict_grade():
    """API endpoint to make predictions based on info from application and credit history."""
    try:
        # Get JSON data from the request and convert it to a data frame
        # NOTE: Data validation step is not implemented.
        df = pd.DataFrame(request.json)

        # Perform predictions using the pipeline (model)
        prediction = pipeline_with_history.predict(df)

        # Probability to default
        probability = pipeline_with_history.predict_proba(df)[:, 1]

        # Return the results as JSON
        return jsonify({
            "model": "Credit default prediction (credit history is needed)",
            "model_version": model_version__with_history,
            "model_output_options": "1: financial difficulties, 0: no financial difficulties",
            "prediction": prediction.tolist(),
            "probability": probability.tolist(),
        })
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    server_port = os.environ.get("PORT", "8080")
    app.run(debug=True, port=server_port, host="0.0.0.0")
