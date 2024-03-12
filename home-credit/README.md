# Home Credit Default Risk Modeling

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
<https://gegznav.github.io/ds-projects/home-credit>   

  </td>
  </tr>
</table>

## Annotation

In this project, an extensive analysis of credit data from **Home Credit Group** was undertaken. Two distinct models were developed to predict if a loan applicant repays the loan or faces financial difficulties: one model operates independently of credit history data and another one incorporates such information. After rigorous testing, the models were successfully deployed on the Google Cloud Platform and now are available via API. Surprisingly, the results indicate only marginal improvement in model performance when historical credit data is included.

## Contents of This Directory

Main:

- `341.ipynb`: 
File with project description *(might not be present on GitHub)*.

- `index.html`:
Rendered report of the analysis **(the main file of this project)**.  
View it via the link provided above.

- `home-credit.ipynb`:
Source code of the data analysis (Jupyter Notebook).


Directories:

- `app`:
Flask app and other files required for model deployment. 
This folder should be treated as a separate project or a separate deployment 
environment.

- `functions`:
Directory for data analysis functions, methods, and classes.

- `img`:
Directory for images and pictures.


Directories *(might not be present on GitHub)*:

- `data`:
Directory for data. It has subdirectories:
    - `info`:
    Directory for metadata and descriptions.
    - `raw`:
    Directory for raw data *(not present on GitHub)*.
    - `interim`:
    Directory for interim data and cached data analysis results *(not present on GitHub)*.

- `models`:
  Directory for models *(not present on GitHub)*.


Files:

- `.gitignore`:
Utility file for Git.

- `requirements.txt`: 
File with a list of Python packages required for this project.
Created using `pip freeze > requirements.txt`.

## Reproducibility

### Working Directory

During the installation and analysis, the working directory of all tools must
 be the root directory of this project 
(i.e., the directory containing the `home-credit.ipynb` file).

### Tools

This project uses Python 3.11 as the main data analysis tool.

To run the analysis, it is recommended to create a separate virtual environment 
(e.g., `proj-home-credit`) 
and install the required Python packages there.
Assuming that [Anaconda](https://www.anaconda.com/download) is installed, this can be accomplished by running the following commands in the terminal:

```bash
conda create -n proj-home-credit python=3.11
conda activate proj-home-credit
pip install -r requirements.txt
```

### HTML Report

To create an HTML report, install [Quarto](https://quarto.org/docs/download/) (version 1.4 or newer is recommended) and run the following command in the terminal:

```bash
quarto render home-credit.ipynb --to html --output index.html
```

### Additional Instructions

Additional instructions for reproducing the analysis are provided in the report.

## Model Deployment

To deploy the models, a Flask application was created. The code needed to deploy the application is available in the sub-directory `app` and more details are available in the README file of that sub-directory.