Travel Insurance Claim Prediction
=================================

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
<https://gegznav.github.io/ds-projects/travel-insurance>   

  </td>
  </tr>
</table>


Annotation and Summary
----------------------

A company in the travel industry is introducing a **travel insurance** package that includes coverage for COVID-19. To determine, which customers would be interested in buying it, the company needs to analyze its database history. Data from approximately 2000 previous customers, who were offered the insurance package in 2019, has been provided for this purpose. The **task** is to *build a predictive model* that can determine whether a customer is likely to purchase the travel insurance package.

In this project, a comprehensive analysis of the **travel insurance** dataset was conducted, delving deep into its intricacies. A sophisticated **classification model** was methodically developed, fine-tuned, and **rigorously evaluated** using advanced machine learning techniques. The chosen model, a **Random Forest** algorithm, exhibited an impressive **accuracy of 82.3%** (a balanced accuracy of **76.7%**). This model effectively predicts whether a customer will opt for the insurance or not, based on the customer's *annual income*, *age*, and the *number of family members*. Notably,  **annual income** emerged as the **most influential feature** in making accurate predictions.



Contents of This Directory
--------------------------

Main:

- `index.html`:
Rendered report of the analysis **(the main file of this project)**.  
View it via the link provided above.

- `travel-insurance.ipynb`:
Source code of the data analysis (Jupyter notebook).


Directories:

- `.saved-results`:  
Directory for cached data analysis results (it may be not visible on GitHub).

- `_config`:
Directory with configuration files for Python package `ydata_profile`.

- `_extensions`:
Directory for Quarto extensions.

- `data`:
Directory for data (it may be not visible on GitHub).

- `functions`:
Directory for data analysis functions, methods and classes.

- `img`:
Directory for images and pictures.


Files:

- `.gitignore`:
Utility file for Git.

- `requirements.txt`: 
File with a list of Python packages required for this project.


Reproducibility
---------------

### Working Directory

During the installation and the analysis, the working directory of all tools must be the root directory of this project 
(i.e., the directory containing the `travel-insurance.ipynb` file).

### Tools

This project uses Python 3.11 as the main data analysis tool.

To run the analysis, it is recommended to create a separate virtual environment 
(e.g., `proj-travel-insurance`) 
and install the required Python packages there.
Assuming that [Anaconda](https://www.anaconda.com/download) is installed, this can be accomplished by running the following commands in the terminal:

```bash
conda create -n proj-travel-insurance python=3.11
conda activate proj-travel-insurance
pip install -r requirements.txt
```


### HTML Report

To create an HTML report, install [Quarto](https://quarto.org/docs/download/) (version 1.4 or newer is recommended) and run the following command in the terminal:

```bash
quarto render travel-insurance.ipynb --to html --output index.html
```
