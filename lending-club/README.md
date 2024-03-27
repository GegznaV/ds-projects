# Lending Club Data Analysis Project

<table width="100%">
  <tr>
  <td width="20%">
  <p align="center">
  
  <img src="img/logo-mini.png">

  </p>
  </td> 
  <td width="80%" align="center">
  
  This directory contains a **machine learning project** by [Vilmantas Gėgžna](https://github.com/GegznaV).  
You should **study the report** available **via this link:**  
<https://gegznav.github.io/ds-projects/lending-club>   

  </td>
  </tr>
</table>

## Annotation

In this project, a comprehensive analysis of **Lending Club** loan data was conducted. Changing over-time trends were identified thus to ensure relevance of the modeling phase, data from the most recent year, 2018, was exclusively utilized. The modeling process included two major **tasks**: **predicting loan application status** (accepted/rejected) and **forecasting key attributes of accepted loans**, including grade, sub-grade, and interest rate.

The development of four distinct models was executed with a thorough approach, addressing challenges such as group imbalance and data size. Rigorous procedures were employed to refine and optimize each model. Subsequently, the most effective models were selected for **deployment** on the **Google Cloud Platform** (GCP).

While the models have been successfully deployed and are currently accessible through an API, ongoing efforts for refinement and enhancement are acknowledged. Continuous improvement remains a priority to ensure the models' accuracy and effectiveness over time.



## Contents of This Directory


Main:


- `index.html`:
Rendered report of the analysis **(the main file of this project)**.  
View it via the link provided above.

- `lending-club.ipynb`:
Source code of the data analysis (Jupyter notebook).


Directories:

- `app`:
Flask app and other files required for model deployment.

- `functions`:
Directory for data analysis functions, methods, and classes.

- `img`:
Directory for images and pictures.


Directories *(not present on GitHub)*:

- `data`:
Directory for data *(not present on GitHub)*. It has subdirectories:
    - `raw`:
    Directory for raw data *(not present on GitHub)*.
    - `interim`:
    Directory for interim data and cached data analysis results *(not present on GitHub)*.


Files:

- `.gitignore`:
Utility file for Git.

- `requirements.txt`:
File with a list of Python packages required for this project.
Created using the tool [`pigar`](https://github.com/damnever/pigar) and manually corrected afterwards.

## Reproducibility
### Working Directory

During the installation and analysis, the working directory of all tools must be the root directory of this project 
(i.e., the directory containing the `lending-club.ipynb` file).

### Tools

This project uses Python 3.11 as the main data analysis tool.

To run the analysis, it is recommended to create a separate virtual environment 
(e.g., `proj-lending-club`) 
and install the required Python packages there.
Assuming that [Anaconda](https://www.anaconda.com/download) is installed, this can be accomplished by running the following commands in the terminal:

```bash
conda create -n proj-lending-club python=3.11
conda activate proj-lending-club
pip install -r requirements.txt
```

### HTML Report

To create an HTML report, install [Quarto](https://quarto.org/docs/download/) (version 1.4 or newer is recommended) and run the following command in the terminal:

```bash
quarto render lending-club.ipynb --to html --output index.html
```

### Additional Instructions

Additional instructions for reproducing the analysis are provided in the report.

## Model Deployment

To deploy the models, a Flask application was created. The code needed to deploy the application is available in the sub-directory `app`. More details are available in the README file of that sub-directory.
