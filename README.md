# Glioma Grade Classification - DSC 201 Project 2

This course project will compare four models for predicting glioma tumor grade:
Logistic Regression, Random Forest, AdaBoost, and XGBoost.

The approved data source is the [UCI Glioma Grading Clinical and Mutation Features dataset](https://archive.ics.uci.edu/dataset/759/glioma-grading-clinical-and-mutation-features-dataset), dataset ID 759.
The target is `Grade`: 0 means Lower-Grade Glioma (LGG), and 1 means Glioblastoma (GBM).

This is an educational model comparison, not a medical diagnostic system.

## Current progress

Steps 1 and 2 are complete: the project structure is in place, and the official
UCI dataset has been downloaded and inspected. There are 839 rows, 23 input
features, no missing values, and one repeated full record. No rows have been
removed. The classes contain 487 LGG and 352 GBM cases.

Plots, preprocessing, model training, and prediction results have not been added yet.

## Setup

Open a terminal in this repository and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In VS Code, select this environment when running Python or the notebook.

## Run the current step

```powershell
.\.venv\Scripts\python.exe main.py
```

This downloads the official UCI dataset, saves it without changing any values,
and prints the inspection checks. Internet access is required. You can also run
[project_notebook.ipynb](./project_notebook.ipynb) from this repository folder;
it uses the same functions and includes short explanations of the findings.

## Project files

- [main.py](./main.py): data loading and inspection functions.
- [project_notebook.ipynb](./project_notebook.ipynb): inspection code and explanations.
- [data/glioma.csv](./data/glioma.csv): unchanged local copy of the 23 features and Grade.
- [data/variable_dictionary.csv](./data/variable_dictionary.csv): UCI variable descriptions.
- [results/](./results/): model results and summary CSV files once experiments run.
- [results/figures/](./results/figures/): saved plots.
- [report/](./report/): the report and presentation outline, to be written after analysis.

## Next step

Create the required exploration plots and review the preprocessing choices,
including the repeated record, age outliers, and text-valued Race feature.

## Dataset citation

Tasci, E., Camphausen, K., Krauze, A. V., and Zhuge, Y. (2022).
*Glioma Grading Clinical and Mutation Features*. UCI Machine Learning Repository.
https://doi.org/10.24432/C5R62J
