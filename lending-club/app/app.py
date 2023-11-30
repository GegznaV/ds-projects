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
model_version__status = "1.0"
pipeline_loan_status = joblib.load("./models/01--model_predict_loan_status.pkl")

model_version__grade = "1.0"
pipeline_loan_grade = joblib.load("./models/02--model_predict_grade.pkl")

model_version__subgrade = "1.0"
pipeline_loan_subgrade = joblib.load("./models/03--model_predict_subgrade.pkl")

model_version__interest_rate = "1.0"
pipeline_loan_interest_rate = joblib.load(
    "./models/04--model_predict_interest_rate.pkl"
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
        model_1_version=model_version__status,  # Loan status (accepted/rejected)
        model_2_version=model_version__grade,  # Loan grade
        model_3_version=model_version__subgrade,  # Loan sub-grade
        model_4_version=model_version__interest_rate,  # Loan interest rate
    )


# To test if the server is up and running
@app.route("/test", methods=["GET"])
def test_server():
    """Test page without html template."""
    return "OK - Server is up and running!"


# To get the model's predictions
@app.route("/api/predict-loan-status", methods=["POST"])
def predict_loan_status():
    """API endpoint to predict loan application status (accepted/rejected)."""
    try:
        # Get JSON data from the request and convert it to a data frame
        # NOTE: Data validation step is not implemented.
        df = pd.DataFrame.from_dict(request.json)

        # Perform predictions using the pipeline (model)
        prediction = pipeline_loan_status.predict(df)

        # Probability of accepted loan applications
        probability = pipeline_loan_status.predict_proba(df)[:, 1]

        # Return the results as JSON
        return jsonify({
            "model": "Loan application status prediction",
            "model_version": model_version__status,
            "model_output_options": "1-accepted, 0-rejected",
            "prediction": prediction.tolist(),
            "probability": probability.tolist(),
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/predict-grade", methods=["POST"])
def predict_grade():
    """API endpoint to predict loan grade (A, B, C, D, E, F, G)."""
    try:
        # Get JSON data from the request and convert it to a data frame
        # NOTE: Data validation step is not implemented.
        df = pd.DataFrame(request.json)

        # Perform predictions using the pipeline (model)
        prediction = pipeline_loan_grade.predict(df)

        # Return the results as JSON
        return jsonify({
            "model": "Loan grade prediction",
            "model_version": model_version__grade,
            "model_output_options": "A, B, C, D, E, F, G",
            "prediction": prediction.tolist(),
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/predict-subgrade", methods=["POST"])
def predict_subgrade():
    """API endpoint to predict loan subgrade (A1, A2, A3, ..., G5)."""
    try:
        # Get JSON data from the request and convert it to a data frame
        # NOTE: Data validation step is not implemented.
        df = pd.DataFrame.from_dict(request.json)

        # Perform predictions using the pipeline (model)
        prediction = pipeline_loan_subgrade.predict(df)

        # Return the results as JSON
        return jsonify({
            "model": "Loan sub-grade prediction",
            "model_version": model_version__subgrade,
            "model_output_options": "A1, A2, ..., A5, B1, B2, ..., G5",
            "prediction": prediction.tolist(),
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/predict-interest-rate", methods=["POST"])
def predict_interest_rate():
    """API endpoint to predict loan interest rate (in percent)"""
    try:
        # Get JSON data from the request and convert it to a data frame
        # NOTE: Data validation step is not implemented.
        df = pd.DataFrame.from_dict(request.json)

        # Perform predictions using the pipeline (model)
        prediction = pipeline_loan_interest_rate.predict(df)

        # Return the results as JSON
        return jsonify({
            "model": "Loan interest rate prediction",
            "model_version": model_version__interest_rate,
            "model_output_units": "percent (%)",
            "prediction": prediction.tolist(),
        })
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    server_port = os.environ.get("PORT", "8080")
    app.run(debug=True, port=server_port, host="0.0.0.0")
