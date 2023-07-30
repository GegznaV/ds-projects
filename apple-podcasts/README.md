The Analysis of *Apple Podcasts* Reviews
========================================

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
<https://gegznav.github.io/ds-projects/apple-podcasts>   

  </td>
  </tr>
</table>


Annotation
----------

A **podcast** is a series of spoken word episodes, all focused on a particular topic or theme, like cycling or startups
(source: [The Podcast Host](https://www.thepodcasthost.com/listening/what-is-a-podcast/)).
It is a program made available in digital format either for download over the Internet
(source: [Wikipedia](https://en.wikipedia.org/wiki/Podcast)) or watching/listening online.
[Apple Podcasts](https://www.apple.com/apple-podcasts/) is one of the prominent platforms dedicated to hosting podcasts.
The comprehensive analysis encompasses a staggering collection of over **100,000 podcasts** hosted on the platform and an impressive tally of more than **2 million reviews**, all produced within a substantial timeframe stretching from **December 9, 2005**, to **December 8, 2022**. This vast dataset provides extensive coverage of approximately **17 years** of podcasting content and feedback. The analysis of it uncovers the most popular and well-rated podcast categories, interesting temporal (daily, weekly and other) patterns as well as relationships between various podcast and review attributes.


Contents of This Directory
--------------------------

- `index.html`:
Rendered report of the analysis **(the main file of this project)**.  
View it via the link provided above.

- `apple-podcasts.ipynb`:
Source code of the data analysis (Jupyter notebook).

- `data`:
Directory for data (it may be not visible on GitHub).

- `img`:
Directory for images and pictures.

- `.gitignore`:
Utility file for Git.

- `functions.py`: 
File with custom Python functions.

- `functions.R`: 
File with custom R functions.

- `requirements.txt`: 
File with a list of Python packages required for this project.


Reproducibility
---------------

### Working Directory

During the installation and analysis, the working directory of all tools must be the root directory of this project 
(i.e., the directory containing the `apple-podcasts.ipynb` file).


### Tools

This project uses Python 3.11 as the main data analysis tool and uses R 4.3.1 to run some scripts.

To run the analysis, it is recommended to create a separate virtual environment 
(e.g., `proj-apple-podcasts`) 
and install the required Python packages there.
Assuming that [Anaconda](https://www.anaconda.com/download) is installed, this can be accomplished by running the following commands in the terminal:

```bash
conda create -n proj-apple-podcasts python=3.11
conda activate proj-apple-podcasts
pip install -r requirements.txt
```

TO work out of the box, it is also expected [program R](https://www.r-project.org/) version 4.3.1 to be installed in "C:/PROGRA~1/R/R-4.3.1" with additional R packages `tibble`,  `dplyr`, `purrr`, `stringr`, `rstatix`, and `multcompView`. To install the packages, the following R code can be used (in R console):

```r
install.packages("tibble")
install.packages("dplyr")
install.packages("purrr")
install.packages("stringr")
install.packages("rstatix")
install.packages("multcompView")
```

If you use a different version of R or have it installed in a different directory, you should change the path to it in both the `functions.py` and `apple-podcasts.ipynb` files.


### HTML Report

To create an HTML report, install [Quarto](https://quarto.org/docs/download/) (version 1.4 or newer is recommended) and run the following command in the terminal:

```bash
quarto render apple-podcasts.ipynb --to html --output index.html
```


Dashboard
--------------

A part of this project was to create a dashboard in *Looker Studio* to visualize the results of the analysis. 

- *Looker Studio* dashboard "Apple Podcasts: Ratings and Amount of Reviews" is available 
  <a href="https://datastudio.google.com/reporting/e10e312b-8ccc-44b9-b85d-984956d496f0" target="_blank">here</a>.
