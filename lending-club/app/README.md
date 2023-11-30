# Flask App to Deploy Predictive Models

This directory contains the files required to create a Flask application with machine learning models based on **Lending Club** data and deploy it on **Google Cloud Platform** (GCP). The application allows accessing 4 different models.
To access the models and other functionality, use the main URL of the app and add the following routes.

- The **main URL** on GCP currently is: <https://lending-club-app-x6jg32rquq-ew.a.run.app>
- The **routes:**
    - Test if the service is up and running: /test 
    - **Model 1** (loan application status prediction): /api/predict-loan-status
    - **Model 2** (grade prediction): /api/predict-grade
    - **Model 3** (sub-grade prediction): /api/predict-subgrade
    - **Model 4** (interest rate prediction): /api/predict-interest-rate
    
You may access the application via an online server or a local development server.
For predictions, you send a request to the application providing the required information as a JSON object (see examples below).  In the examples, command line tool `curl` will be ued. The **data** used in the examples are present in `test-data/` directory.


### Predictions via Online Server

To run the examples, command line tool `curl` must be installed.

- Test if the service is up and running:

```bash
curl -k https://lending-club-app-x6jg32rquq-ew.a.run.app/test
```

Response:

```
OK - server is up and running!
```

- **Predict loan application status (accepted or rejected):**


```bash
curl https://lending-club-app-x6jg32rquq-ew.a.run.app/api/predict-loan-status --ssl-no-revoke \
     -H 'Content-Type: application/json' \
     -d @test-data/data-to-predict-status.json
```

Response:

```json
{
  "model": "Loan application status prediction",
  "model_output_options": "1-accepted, 0-rejected",
  "model_version": "1.0",
  "prediction": [
    0.0
  ],
  "probability": [
    2.1901731041585664e-05
  ]
}
```

Loan application status prediction API needs pre-processed data with at least the variables that are present in `test-data/data-to-predict-status.json` file.


- **Predict loan grade:**

```bash
curl https://lending-club-app-x6jg32rquq-ew.a.run.app/api/predict-grade --ssl-no-revoke \
     -H 'Content-Type: application/json' \
     -d @test-data/data-to-predict-grade.json
```

Response:

```json
{
  "model": "Loan grade prediction",
  "model_output_options": "A, B, C, D, E, F, G",
  "model_version": "1.0",
  "prediction": [
    "G",
    "A",
    "C"
  ]
}
```

API needs data in the form provided by Lending Club.
There should be at least these variables as in `test-data/data-to-predict-grade.json` file.


- **Predict loan sub-grade:**

```bash
curl  https://lending-club-app-x6jg32rquq-ew.a.run.app/api/predict-subgrade --ssl-no-revoke \
     -H 'Content-Type: application/json' \
     -d @test-data/data-to-predict-subgrade.json
```

Response:

```json
{
  "model": "Loan sub-grade prediction",
  "model_output_options": "A1, A2, ..., A5, B1, B2, ..., G5",
  "model_version": "1.0",
  "prediction": [
    "B1",
    "A3",
    "C2"
  ]
}
```

API needs data in the form provided by Lending Club.
There should be at least these variables as in `test-data/data-to-predict-interest-rate.json` file.


- **Predict interest rate:**

```bash
curl  https://lending-club-app-x6jg32rquq-ew.a.run.app/api/predict-interest-rate --ssl-no-revoke \
     -H 'Content-Type: application/json' \
     -d @test-data/data-to-predict-interest-rate.json
```

Response:

```json
{
  "model": "Loan interest rate prediction",
  "model_output_units": "percent (%)",
  "model_version": "1.0",
  "prediction": [
    28.845358436592004,
    6.799605309001053,
    15.528215704748181
  ]
}
```

API needs data in the form provided by Lending Club.
There should be at least these variables as in `test-data/data-to-predict-interest-rate.json` file.


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
curl -X POST http://localhost:8080/api/predict-loan-status \
     -H 'Content-Type: application/json' \
     -d @test-data/data-to-predict-status.json
```

```bash
curl -X POST http://localhost:8080/api/predict-grade --ssl-no-revoke \
     -H 'Content-Type: application/json' \
     -d @test-data/data-to-predict-grade.json
```

