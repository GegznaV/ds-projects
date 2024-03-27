# Classification of Toxic Comments

<table width="100%">
  <tr>
  <td width="20%">
  <p align="center">
  
  <img src="img/logo-mini.png">

  </p>
  </td> 
  <td width="80%" align="center">
  
  This directory contains a **deep learning project** by [Vilmantas Gėgžna](https://github.com/GegznaV).  
You should **study the report** available **via this link:**  
<https://gegznav.github.io/ds-projects/toxic-comments>   

  </td>
  </tr>
</table>

## Annotation

In response to the challenges posed by online abuse and harassment hindering open discourse, a data science project was undertaken to develop an effective moderation system. The project utilized a **DistillBERT**-based model for **multi-label text classification**, leveraging transfer learning techniques for **deep learning**. Built using **PyTorch and Lightning** frameworks, the model was trained on a dataset comprising **223,549 comments**, encompassing both clean (non-toxic) and various forms of toxic content (toxic, severe toxic, obscene, threat, insult, identity hate). Rigorous state-of-the-art procedures were employed to ensure robustness and accuracy. The model achieved a balanced **accuracy** of **0.964** and an **F1 score** of **0.551** on the test set, demonstrating its efficacy in identifying and moderating potentially harmful comments. This project aims to contribute to the creation of safer and more inclusive online communities by facilitating constructive conversations while mitigating the impact of abusive behavior.

## Contents of This Directory

Main:

- `index.html`:
Rendered report of the analysis **(the main file of this project)**.  
View it via the link provided above.

- `toxic-comments.ipynb`:
Source code of the data analysis (Jupyter Notebook).


Directories:

- `img`:
Directory for images and pictures.


Directories *(might not be present on GitHub)*:

- `data`:
Directory for data. It has subdirectories:
    - `raw`:
    Directory for the raw data.
    - `processed`:
    Directory for pre-processed data and analysis results.

- `logs`:
  Directory for logs and checkpoints.


Files:

- `.gitignore`:
Utility file for Git.

- `requirements.txt`: 
File with a list of Python packages required for this project on a Windows-based environment.
Created using the tool [`pigar`](https://github.com/damnever/pigar) and manually corrected afterwards. *NOTE:* package versions used in Colab might differ.

## Reproducibility

### Working Directory

During the installation and analysis, the working directory of all tools must
 be the root directory of this project 
(i.e., the directory containing the `toxic-comments.ipynb` file).

### Tools

This project uses Python 3.11 in a Windows-based environment (local) and 3.10 in Colab.

To run the analysis, it is recommended to create a separate virtual environment 
(e.g., `proj-toxic-comments`) and install the required Python packages there.
Assuming that [Anaconda](https://www.anaconda.com/download) is installed, this can be accomplished by running the following commands in the terminal:

```bash
conda create -n proj-toxic-comments python=3.11
conda activate proj-toxic-comments
pip install -r requirements.txt
```

To install `pythorch` with CUDA (v12.1 or higher) support, run the following command in the terminal:
```bash
pip install "torch==2.1.2+cu121"  "torchvision==0.16.2+cu121" \
  --index-url https://download.pytorch.org/whl/cu121
```

If you require support for different CUDA versions, please refer to the PyTorch installation [website](https://pytorch.org/get-started/locally/) for more details.

### HTML Report

To create an HTML report, install [Quarto](https://quarto.org/docs/download/) (version 1.4 or newer is recommended) and run the following command in the terminal:

```bash
quarto render toxic-comments.ipynb --to html --output index.html
```
