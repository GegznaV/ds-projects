# Stroke Risk Prediction

<table width="100%">
  <tr>
  <td width="20%">
  <p align="center">
  
  <img src="img/logo-mini.png">

  </p>
  </td> 
  <td width="80%" align="center">
  
  This directory contains a **data analysis project** by [Vilmantas Gėgžna](https://github.com/GegznaV).  
You should **study the report** available **via this link:**  
<https://gegznav.github.io/ds-projects/stroke-prediction>   

  </td>
  </tr>
</table>

## Annotation


According to the World Health Organization (WHO), **stroke** stands as the **leading cause of disability** and the **second leading** cause **of death** on a global scale. Consequently, tools aimed at anticipating it in advance hold significant potential for stroke prevention. This project was devoted to the thorough analysis of a stroke prediction dataset. It encompassed **exploratory data analysis**, **feature engineering**, and the application of **predictive modeling** techniques to delve deeper into this critical issue and construct an effective predictive model. After careful evaluation, the best-performing model, with an **F1** score of **32.1%**, **balanced accuracy** of **73.7%**, and ROC **AUC** of **0.801**, was chosen and subsequently **deployed in a cloud-based environment**. For comprehensive insights and detailed findings, please refer to the project's report.


## Disclaimer

*This project **does not provide any medical advice**; it is solely for educational and research purposes. If you require medical advice, please consult your physician.*


## Contents of This Directory


Main:

- `325.ipynb`  
File with project description.

- `index.html`:
Rendered report of the analysis **(the main file of this project)**.  
View it via the link provided above.

- `stroke-prediction.ipynb`:
Source code of the data analysis (Jupyter notebook).


Directories:

- `_extensions`:
Directory for Quarto extensions.

- `functions`:
Directory for data analysis functions, methods and classes.

- `img`:
Directory for images and pictures.


Directories *(not present on GitHub)*:

- `.saved-results`:  
Directory for cached data analysis results *(not present on GitHub)*.

- `data`:
Directory for data *(not present on GitHub)*.

- `model_to_deploy`:
Directory for the final model selected for deployment *(not present on GitHub)*.


Files:

- `.gitignore`:
Utility file for Git.

- `requirements.txt`: 
File with a list of Python packages required for this project.


## Reproducibility
### Working Directory

During the installation and the analysis, the working directory of all tools must be the root directory of this project 
(i.e., the directory containing the `stroke-prediction.ipynb` file).

### Tools

This project uses Python 3.11 as the main data analysis tool.

To run the analysis, it is recommended to create a separate virtual environment 
(e.g., `proj-stroke-prediction`) 
and install the required Python packages there.
Assuming that [Anaconda](https://www.anaconda.com/download) is installed, this can be accomplished by running the following commands in the terminal:

```bash
conda create -n proj-stroke-prediction python=3.11
conda activate proj-stroke-prediction
pip install -r requirements.txt
```

To run the analysis, additional tools (such as GraphViz) might be required to install separately.

### HTML Report

To create an HTML report, install [Quarto](https://quarto.org/docs/download/) (version 1.4 or newer is recommended) and run the following command in the terminal:

```bash
quarto render stroke-prediction.ipynb --to html --output index.html
```

### Additional Instructions

Additional instructions for reproducing the analysis are provided in the report.


## Model Deployment

To deploy the model, a Flask application was created. The code needed to deploy the application is available in Github repository [GegznaV/deploy-stroke-prediction](https://github.com/GegznaV/deploy-stroke-prediction). 
For predictions, you send a request to the application providing the following information about the patient as a JSON object:

- `age` (in years);
- `health_risk_score` (integer from 0 to 5);
- `smoking_status` (one of the following: `never smoked`, `formerly smoked`, `smokes`, `Unknown`).

You may access the application via online server or use a local development server.

### Predictions via Online Server

Model is deployed on Render.com and accessible via URL <https://stroke-prediction-af8t.onrender.com>. 
You may test if server is up via route `/test` and make predictions via route `/api/predict`. 


The examples to test the service will use `curl` command line tool.
To to run the examples `curl` must be installed.


```bash
curl -k https://stroke-prediction-af8t.onrender.com/test
```
Response:
```
OK - server is up and running!(proj-stroke-prediction) 
```


To make predictions use, e.g.:
```bash
curl https://stroke-prediction-af8t.onrender.com/api/predict --ssl-no-revoke \
     -H 'Content-Type: application/json' \
     -d '{"age":[30], "health_risk_score":[1], "smoking_status":["never smoked"]}'
```
Response (I formatted it for better readability):
```
{"inputs":{
  "age":[30],
  "health_risk_score":[1],
  "smoking_status":["never smoked"]
  },
  "prediction":[0],
  "stroke_probability":[0.028515656235292997]
}
```

To request predictions about several people at once use, e.g.:
```bash
curl https://stroke-prediction-af8t.onrender.com/api/predict --ssl-no-revoke \
     -H 'Content-Type: application/json' \
     -d '{"age":[30, 65, 84], "health_risk_score":[1, 0, 3], "smoking_status":["never smoked", "smokes", "never smoked"]}'
```
Response (again manually formatted):
```
{"inputs":{
  "age":[30,65,84],
  "health_risk_score":[1,0,3],
  "smoking_status":["never smoked","smokes","never smoked"]
  },
  "prediction":[0,0,1],
  "stroke_probability":[0.028515656235292997,0.05356717176142374,0.7776111298079225]
}
```

### Predictions Locally via Development Server

To deploy app locally and ant test its responses, download the contents of GitHub repository (see above) to your working directory and run the following commands in the terminal:

```bash
python stroke_prediction_app.py
```

To test if the server is running, use:
```bash
curl -X GET http://127.0.0.1:5000/test
```

To make predictions, use, e.g.:
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H 'Content-Type: application/json' \
     -d '{"age":[30], "health_risk_score":[1], "smoking_status":["never smoked"]}'
```

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H 'Content-Type: application/json' \
     -d '{"age":[30, 65, 84], "health_risk_score":[1, 0, 3], "smoking_status":["never smoked", "smokes", "never smoked"]}'
```