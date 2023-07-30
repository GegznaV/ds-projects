Red Wine Quality Prediction
===========================

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
<https://gegznav.github.io/ds-projects/red-wine-quality>   

  </td>
  </tr>
</table>


Annotation
----------

Wine quality assessment and certification are important for **wine making and selling** processes. Certification helps to **prevent counterfeiting** and in this way **protects** people's **health** as well as **assures quality** for the wine market. **To identify the most influential factors** is crucial for effective quality evaluation while to classify wines into quality groups is useful **for setting prices**[^cortez2009].

*Vinho verde* is wine from the Minho region, which is located in the northwest part of Portugal.
In this project, data of **red *vinho verde*** wine specimens are investigated.
The dataset was originally provided by Cortez et al.[^cortez2009]. 
This project **mainly focuses** on:

a) **prediction quality** (to investigate how accurately **(1)** *wine quality* and **(2)** *alcohol content* in wine can be predicted) and
b) **explanation** (to identify, which are the most important factors for the prediction tasks).

<div style="font-size:14px">

[^cortez2009]: Cortez, Paulo, António Cerdeira, Fernando Almeida, Telmo Matos, and José Reis. 2009. “Modeling Wine Preferences by Data Mining from Physicochemical Properties.” Decision Support Systems 47 (4): 547–53. https://doi.org/10.1016/j.dss.2009.05.016

</div>


Contents of This Directory
--------------------------

- `index.html`:
Rendered report of the analysis **(the main file of this project)**.  
View it via the link provided above.

- `red-wine-quality.qmd`: 
Source code of the data analysis (Quarto notebook).

- `assets`:
Directory that contains file with bibliography and related links.

- `data`:
Directory for data and related links.

- `img`:
Directory for images and pictures.

- `_extensions`:
Directory for Quarto extensions.

- `.gitignore`:
Utility file for Git.

- `pyproject.toml`:
File with the configuration used by Quarto filter `black-formatter`.

- `requirements.txt`: 
File with the list of Python packages required for this project.

- `project-RStudio.Rproj`: 
File with RStudio project configuration.

Reproducibility
---------------

### Working Directory

During the installation and analysis, the working directory of all tools must be the root directory of this project 
(i.e., the directory containing the `red-wine-quality.qmd` file).


### Tools


This project uses both R 4.3.1 and Python 3.11 as the main data analysis tools.

1) Programs [R](https://www.r-project.org/) (version 3.4.1) and RStudio (the newest version) should be installed.
  
2)  Next, install the R package `renv`.  To install it, the following R code can be used (in the R console):
``` r
install.packages("renv")
```

3) To install the remaining required R packages, run the following command in the R console and wait until the process ends:
```r
renv::restore()
```

4) To run the Python code, it is recommended to create a separate virtual environment (e.g., `renv/conda-env`) and install the required Python packages there.
Assuming that [Anaconda](https://www.anaconda.com/download) is installed, this can be accomplished by running the following commands in the terminal:

```bash
conda create -p renv/conda-env python=3.11
conda activate renv/conda-env
pip install -r requirements.txt
```


5) Then configure RStudio to use this virtual environment

```r
renv::use_python(type = 'conda', name = 'renv/conda-env')
```


<!--
This project uses both R 4.3.1 and Python 3.11 as the main data analysis tools.

Programs [R](https://www.r-project.org/) (required) and RStudio should be installed as well as R packages `renv`, `tidyverse`, `reticulate`, `factoextra`, `DescTools`, `patchwork`, `knitr`, `pandoc`, `ggstatsplot`, and `rmarkdown` as well as their dependencies. To install the packages, the following R code can be used (in the R console):

``` r
install.packages("tidyverse")
install.packages("renv")
install.packages("reticulate")
install.packages("factoextra")
install.packages("DescTools")
install.packages("patchwork")
install.packages("knitr")
install.packages("pandoc")
install.packages("ggstatsplot")
install.packages("rmarkdown")
```

To work properly, RStudio might ask to install some additional packages.

To run the Python code, it is recommended to create a separate virtual environment (e.g., `proj-red-wine-quality`) and install the required Python packages there.
Assuming that [Anaconda](https://www.anaconda.com/download) is installed, this can be accomplished by running the following commands in the terminal:

```bash
conda create -n proj-red-wine-quality python=3.11
conda activate proj-red-wine-quality
pip install -r requirements.txt
```

Then configure RStudio to use this virtual environment (see the section "Selecting a Default Version of Python" in the ["Using Python with the RStudio IDE"](https://support.posit.co/hc/en-us/articles/1500007929061-Using-Python-with-the-RStudio-IDE) tutorial.)

-->

### HTML Report

To create an HTML report, install [Quarto](https://quarto.org/docs/download/) (version 1.4 or newer is recommended) and run the following command in the terminal:

```bash
quarto render red-wine-quality.qmd --to html --output index.html
```

You may use the „Render“ button in RStudio. In this case, your output document will be called `red-wine-quality.html`.
 

Dashboard
--------------

A part of this project was to create a dashboard in *Looker Studio* to visualize the results of the analysis. 

- *Looker Studio* dashboard "Apple Podcasts: Ratings and Amount of Reviews" is available 
  <a href="https://lookerstudio.google.com/reporting/1413c256-c42a-4b0d-976e-ac2b878fbcf9/page/dFTED" target="_blank">here</a>.
