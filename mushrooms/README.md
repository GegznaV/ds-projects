# Mushrooms Classification: Common Genus's Images

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
<https://gegznav.github.io/ds-projects/mushrooms>   

  </td>
  </tr>
</table>

## Annotation

This project addresses the pressing **issue of mushroom poisoning**, with 7,500 annual cases reported in the United States, mainly due to misidentification. The initiative focuses on developing a **deep learning-based model for mushroom classification**, leveraging advanced techniques like computer vision, data augmentation, and transfer learning. Employing tools such as **Python**, **PyTorch**, **Lightning**, and **TensorBoard**, the project executes a rigorous plan to create four models based on the ***ResNet-18*** architecture, pre-trained on ***ImageNet***. The best-performing model achieves remarkable **accuracy**, with **82.1%** on the test set and an efficient **prediction speed** of **18 milliseconds per image**, showcasing practicality **for real-time applications**. A genus-specific analysis identifies *Suillus*, *Amanita*, and *Boletus* among the most reliably predicted genera, yet, 10.8% of predictions of edible _Agaricus_ are truly poisonous _Amanita_ raises concerns about finding room for model improvements. Further investigation emphasizes the potential benefits of image standardization, prompting **considerations for refining data acquisition strategies** to optimize the model's predictive capabilities.


## Contents of This Directory

Main:

- `415.ipynb`: 
File with project description *(might not be present on GitHub)*.

- `index.html`:
Rendered report of the analysis **(the main file of this project)**.  
View it via the link provided above.

- `mushrooms.ipynb`:
Source code of the data analysis (Jupyter Notebook).


Directories:

- `img`:
Directory for images and pictures.


Directories *(might not be present on GitHub)*:

- `data`:
Directory for data. It has subdirectories:
    - `raw_ok`:
    Directory for raw data without errors and irrelevant images *(not present on GitHub)*.
    - `resized_256`:
    Directory for resized images *(not present on GitHub)*.

- `logs`:
  Directory for logs, profiling data and checkpoints *(not present on GitHub)*.


Files:

- `.gitignore`:
Utility file for Git.

- `requirements.txt`: 
File with a list of Python packages required for this project.
Created using the tool [`pigar`](https://github.com/damnever/pigar).

## Reproducibility

### Working Directory

During the installation and analysis, the working directory of all tools must
 be the root directory of this project 
(i.e., the directory containing the `mushrooms.ipynb` file).

### Tools

This project uses Python 3.11 as the main data analysis tool.

To run the analysis, it is recommended to create a separate virtual environment 
(e.g., `proj-mushrooms`) and install the required Python packages there.
Assuming that [Anaconda](https://www.anaconda.com/download) is installed, this can be accomplished by running the following commands in the terminal:

```bash
conda create -n proj-mushrooms python=3.11
conda activate proj-mushrooms
pip install -r requirements.txt
```

To install `pythorch` with CUDA support, run the following command in the terminal:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

More details on installing `pythorch` with CUDA support can be found [here](https://pytorch.org/get-started/locally/).

### HTML Report

To create an HTML report, install [Quarto](https://quarto.org/docs/download/) (version 1.4 or newer is recommended) and run the following command in the terminal:

```bash
quarto render mushrooms.ipynb --to html --output index.html
```
