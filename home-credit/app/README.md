# Flask App to Deploy Predictive Models

This directory contains the files required to create a Flask application with machine learning models based on **Home Credit Group** data and deploy it on **Google Cloud Platform** (GCP). The application allows accessing 2 different models.
To access the models and other functionality, use the main URL of the app and add the following routes.

- The **main URL** on GCP currently is: <https://home-credit-default-prediction-sarhiiybua-ew.a.run.app>
- The **routes:**
    - Test if the service is up and running: /test 
    - **Model 1** (does not require credit history): /api/predict
    - **Model 2** (requires credit history data): /api/predict-with-credit-history
    
You may access the application via an online server or a local development server.
For predictions, you send a request to the application providing the required information as a JSON object (see examples below).  In the examples, the command line tool `curl` will be used. The **data** used in the examples are present in `test-data/` directory.



### Predictions via Online Server

To run the examples, the command line tool `curl` must be installed.

- Test if the service is up and running:

```bash
curl -k https://home-credit-default-prediction-sarhiiybua-ew.a.run.app/test
```

Response:

```
OK - server is up and running!
```

- **Predict home credit default (no credit history data is required):**

```bash
curl https://home-credit-default-prediction-sarhiiybua-ew.a.run.app/api/predict --ssl-no-revoke \
     -H 'Content-Type: application/json' \
     -d @test-data/data-without-credit-history.json
```

Response:

```json
{
  "model": "Credit default prediction (no credit history is needed)",
  "model_output_options": "1: financial difficulties, 0: no financial difficulties",
  "model_version": "1.0",
  "prediction": [
    0,
    0
  ],
  "probability": [
    0.4116794310847024,
    0.3414021721487845
  ]
}
```

**Note:** this model needs pre-processed data with the exact variables that 
are present in `test-data/data-without-credit-history.json` file.



- **Predict home credit default (credit history data is required):**

```bash
curl https://home-credit-default-prediction-sarhiiybua-ew.a.run.app/api/predict-with-credit-history --ssl-no-revoke \
     -H 'Content-Type: application/json' \
     -d @test-data/data-with-credit-history.json
```

Response:

```json
{
  "model": "Credit default prediction (credit history is needed)",
  "model_output_options": "1: financial difficulties, 0: no financial difficulties",
  "model_version": "1.0",
  "prediction": [
    0,
    0
  ],
  "probability": [
    0.37757032172077054,
    0.2063442279400693
  ]
}
```

**Note:** this model needs pre-processed data with exact variables that 
are present in `test-data/data-with-credit-history.json` file.



### Predictions Locally via Development Server

To run the app locally, set the directory with this file as the working directory, install Python and the required dependencies (it is recommended to have a separate virtual environment for this), and use the following command in the terminal (tested on Windows 10):

```bash
python app.py
```

To test if the server is running, use:
```bash
curl -X GET http://localhost:8080/test
```

To make predictions, use, e.g.:

```bash
curl -X POST http://localhost:8080/api/predict \
     -H 'Content-Type: application/json' \
     -d @test-data/data-without-credit-history.json
```

```bash
curl -X POST http://localhost:8080/api/predict-with-credit-history --ssl-no-revoke \
     -H 'Content-Type: application/json' \
     -d @test-data/data-with-credit-history.json
```
