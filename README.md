# Glioma Grade Classification - DSC 201 Project 2

This course project will compare four models for predicting glioma tumor grade:
Logistic Regression, Random Forest, AdaBoost, and XGBoost.

The approved data source is the [UCI Glioma Grading Clinical and Mutation Features dataset](https://archive.ics.uci.edu/dataset/759/glioma-grading-clinical-and-mutation-features-dataset), dataset ID 759.
The target is `Grade`: 0 means Lower-Grade Glioma (LGG), and 1 means Glioblastoma (GBM).

This is an educational model comparison, not a medical diagnostic system.

## Current progress

The project includes data inspection, exploration plots, basic preprocessing,
and implementations of all four required models. All 30 repetitions have run
for both feature sets, and the notebook, report, and presentation outline use
the completed results.
The original UCI copy has 839 rows and 23 input features, with no missing values.
It contains 487 LGG and 352 GBM cases. This original CSV is kept unchanged.

Preprocessing keeps only the first copy of one exact repeated record, leaving
838 rows (486 LGG and 352 GBM). This prevents identical records appearing in both
split sets, but does not establish that they were the same patient. All ages
are retained: none were flagged by the 1.5-times-IQR check. Race is encoded as
four yes/no columns, giving 26 numeric model inputs.

The seed 42 example split has 670 training rows and 168 test rows. The split
function supports optional training-only standardization; later cross-validation
will instead put scaling inside model pipelines to protect validation folds.
Each model is tuned by five-fold stratified GridSearchCV inside the training
data. GBM (1) is the positive class, and F1 is the tuning score. Seeds 100 through
129 give the 30 required repetitions. All models use the same split and folds
within each repetition. Logistic Regression is scaled inside its pipeline;
tree models are not scaled. The best model from each search is automatically
refitted on that repetition's complete training set.

The optional comparison repeats the same procedure using only the 20 recorded
mutation features. This gives 240 model runs: 30 splits x 4 models x 2 feature sets.

## Main results

All-feature held-out test scores averaged over 30 repetitions:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.8706 | 0.8078 | 0.9131 | 0.8566 |
| Random Forest | 0.8655 | 0.7996 | 0.9117 | 0.8514 |
| AdaBoost | 0.8633 | 0.7860 | 0.9319 | 0.8523 |
| XGBoost | 0.8641 | 0.7916 | 0.9230 | 0.8517 |

Logistic Regression has the highest mean F1 in this configuration, with a small
lead. Adding clinical features increases mean F1 only for AdaBoost, and that
increase is small. See the report for SD, ranges, training/testing comparisons,
and limits on what these results establish.

## Setup

Open a terminal in this repository and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In VS Code, select this environment when running Python or the notebook.

## Run the full project

```powershell
.\.venv\Scripts\python.exe main.py
```

This loads the saved UCI copy (or downloads it if missing), prints inspection
checks, creates six exploration plots, prepares the data, runs all experiments,
and saves the result tables and five model-comparison figures. Internet access is
needed only for a download. To download a fresh copy, call `load_data(refresh=True)`.
The complete experiment can take several minutes. Progress is printed for every
model, and the results CSV is saved after each complete repetition. Wait for all
30 repetitions before using it as a final result table.

In a regular Python run, close each plot window to continue to the next chart.
For a noninteractive run that only saves images:

```powershell
$env:MPLBACKEND = 'Agg'
.\.venv\Scripts\python.exe main.py
```

Matplotlib may print that this backend cannot show a figure; the PNGs are still
saved. Omit that environment setting to use interactive plot windows.
You can also run
[project_notebook.ipynb](./project_notebook.ipynb) from this repository folder;
it uses the same functions and includes short explanations of the findings.
Its experiment cell loads completed CSV results by default. Set
`RUN_EXPERIMENTS = True` there to rerun all model searches from the notebook.
If the result CSVs do not exist yet, run the full script first or enable that setting.

## Project files

- [main.py](./main.py): shared data, plotting, model, tuning, and evaluation functions.
- [project_notebook.ipynb](./project_notebook.ipynb): executed code, plots, and explanations.
- [data/glioma.csv](./data/glioma.csv): unchanged local copy of the 23 features and Grade.
- [data/variable_dictionary.csv](./data/variable_dictionary.csv): UCI variable descriptions.
- [results/model_results.csv](./results/model_results.csv): individual test and training scores, CV F1, seeds, and best parameters for each run.
- [results/summary_results.csv](./results/summary_results.csv): test mean, sample SD, minimum, and maximum by feature set, model, and metric.
- [results/](./results/): also includes each main model's actual and predicted classes for seed 100.
- [results/figures/](./results/figures/): six exploration plots and five model-comparison figures.
- [report/project_report.md](./report/project_report.md): written report based on the saved results.
- [report/presentation_outline.md](./report/presentation_outline.md): seven-slide outline with speaker notes for about six minutes.
- [report/project_report.pdf](./report/project_report.pdf): formatted report with tables, figures, and the UCI citation.
- [report/presentation.pptx](./report/presentation.pptx): editable seven-slide PowerPoint with speaker notes.
- [report/presentation.pdf](./report/presentation.pdf): PDF copy of the slides for easy viewing.

## Presentation and submission

The PDF report and PowerPoint are ready for review. Open the PowerPoint's Notes
pane or Presenter View to see the speaker notes. The suggested timings in the
outline total about six minutes; practice once aloud before presenting.

Use the report PDF, PowerPoint, and code repository as required by your course's
submission page. The supplied guidelines do not specify an LMS upload procedure.
The presentation PDF is an optional viewing copy, not a replacement for the
editable PowerPoint. If you edit the Markdown report or change the results,
update the exported documents too.

## Interpretation and reproducibility

The 30 test groups overlap, so their SD describes changes across splits, not
30 independent studies or a confidence interval. Confusion matrices show only
replicate 1 (seed 100), selected in advance. Reported precision, recall, and F1
refer to GBM, not macro averages. Training scores have a separate `Training_`
prefix and are never labeled as test scores.

The main library versions used for the experiments are listed in
[requirements.txt](./requirements.txt). Python 3.13 was used for verification.
The report describes this saved configuration. If you change data, grids, or
seeds and rerun, review the report and speaker notes against the new results.

No external patient data has been used for validation. These results support an
educational comparison, not clinical deployment or claims about individual patients.

## Verification completed

- The full script finished all 240 runs and saved 11 figures.
- The notebook executed with saved experiment results, showing all 11 figures
  without cell errors.
- All 30 seeds and every selected parameter combination were checked.
- All 32 summary rows were checked against the individual-run CSV, including
  sample SD (`ddof=1`), minimum, and maximum.
- Seed-100 prediction files reproduce their recorded metrics and use identical
  test records across all four models.
- The split and optional standardization were checked for row separation,
  alignment, and use of training-only scaling values.
- The 14-page report PDF and seven-slide PowerPoint were rendered and visually
  checked. Every slide includes speaker notes; a slide PDF is also provided.

## Dataset citation

Tasci, E., Camphausen, K., Krauze, A. V., and Zhuge, Y. (2022).
*Glioma Grading Clinical and Mutation Features*. UCI Machine Learning Repository.
https://doi.org/10.24432/C5R62J
