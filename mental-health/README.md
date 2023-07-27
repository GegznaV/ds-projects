The Analysis of Mental Health in Tech Industry
===============================================

<table width="100%">
  <tr>
  <td width="20%">
  <p align="center">
  
  <img src="img/logo-mini.png">

  </p>
  </td> 
  <td width="80%" align="center">
  
  This directory contains **data analysis project** by [Vilmantas Gėgžna](https://github.com/GegznaV).  
**You should study the rendered report available via [this link](https://gegznav.github.io/ds-projects/mental-health)!** 

  </td>
  </tr>
</table>




Annotation
----------

Various mental disorders are a **widely spread** phenomenon: approximately **1 in 5 adults** (21%, 52.9 million people) in the US experienced mental illness in 2020[^1] and up to 1 billion people across the globe suffer from various mental disorders.
**Loss in productivity** due to anxiety and depression — two most common conditions — alone **costs 1 trillion US dollars** annually worldwide[^2]. 
Still, not enough attention is paid to mental health and not enough healthcare resources are assigned to improve the situation.
As one of the solutions in the US, Open Sourcing Mental Health a non-profit corporation was founded in 2013.
The purpose is to raise awareness, educate, and provide resources to support mental wellness in the tech and open source communities[^3].
Since 2014, the corporation organizes surveys to investigate and understand the status of mental health as well as attitudes towards mental health and frequency of mental health disorders in the tech industry.
This project is dedicated to the analysis of these surveys' data, acquired in 2014-2019. 


<div style="font-size:14px">

[^1]: National Alliance of Mental Illness. URL: <https://www.nami.org/mhstats> (updated on June 2022) 
[^2]: Mental health matters. The Lancet Global Health, 2020. URL: <https://www.thelancet.com/journals/langlo/article/PIIS2214-109X(20)30432-0/fulltext>, DOI: <https://doi.org/10.1016/S2214-109X(20)30432-0> 
[^3]: Open Sourcing Mental Health (OSMI) website. URL: <https://osmihelp.org/about/about-osmi> (visited on 2022-11-15) 

</div>

Contents of This Directory
---------------------------

- `index.html`:
Rendered report of the analysis. View it via the link provided above.  
**This is the main file of this project**.

- `mental-health.ipynb`:
Source code of the data analysis (Jupyter notebook).

- `functions.py`:
Custom Python functions.

- `data`:
Directory for data. It contains SQLite database and the link to its source website.

- `img`:
Directory for images and pictures.

- `requirements.txt`:
List of Python packages required to run the code.

- `.gitignore`:
Utility file for Git.

- Supplementary files:
    - `supplement-a--template.ipynb`:
    Jupyter notebook with the code to investigate each question from the database and the answers to that question.
    - `supplement-b--common-questions.ipynb`:
    Jupyter notebook with the code that lists the questions, which are common among several surveys stored within the database. 


Install Requirements
--------------------

To run the analysis, it is recommended to create a separate virtual environment (e.g., `proj-mental-health`) and install the required Python packages there.
Assuming that [Anaconda](https://www.anaconda.com/download) is installed, this can be accomplished by running the following commands in the terminal:

```bash
conda create -n proj-mental-health python=3.11
conda activate proj-mental-health
pip install -r requirements.txt
```


Create HTML Report
------------------

To create the HTML report, install [Quarto](https://quarto.org/docs/download/) (version 1.4 or newer is recommended) and run the following command in the terminal:

```bash
quarto render mental-health.ipynb --to html --output index.html
```
