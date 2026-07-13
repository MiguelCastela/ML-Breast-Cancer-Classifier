# Breast Cancer Coimbra Classification

A machine learning pipeline that classifies patients as Healthy or Cancer using
routine blood-analysis measurements, built for the Machine Learning
(Aprendizagem Computacional) course. The project covers the full workflow:
data loading and standardization, statistical feature selection, dimensionality
reduction, a wide set of classifiers, and hyperparameter tuning, with all metrics
averaged over multiple randomized runs.

## Dataset

The project uses the Breast Cancer Coimbra dataset (`dataR2`), 116 instances with
9 numeric features and a binary label:

- Features: Age, BMI, Glucose, Insulin, HOMA, Leptin, Adiponectin, Resistin, MCP.1
- Label (`Classification`): 1 = Healthy (52 instances), 2 = Cancer (64 instances)

The raw data lives in `src/data/` as both `dataR2.csv` and `dataR2.xlsx`. The
loader reads the Excel file, drops rows with missing values, relabels the classes
as `Healthy` and `Cancer`, and performs a stratified 70/15/15 train/validation/test
split. All features are standardized using the training-set mean and standard
deviation, and those same statistics are applied to the validation and test sets.

## Pipeline

The stages, orchestrated by `src/main.py`:

1. Data loading and standardization (`preprocessing/`)
2. Feature selection (`feature_selection/`)
   - ROC-AUC ranking per feature
   - Kruskal-Wallis H-test ranking per feature
   - Correlation matrix of the top features
3. Dimensionality reduction (`dimensionality_reduction/`)
   - PCA with scree plot and Kaiser criterion
   - LDA projection onto the first discriminant
4. Classification (`classifiers/`) over four feature sets: all features, the
   top-5 ROC-AUC features, the top-5 Kruskal-Wallis features, the PCA components,
   and the LDA projection
5. Hyperparameter search with stratified k-fold cross-validation (`tuning/`)

Classifiers included: Euclidean minimum-distance, Mahalanobis minimum-distance,
Fisher LDA, SVM (linear, custom, and RBF kernels), k-Nearest Neighbors, Decision
Tree, Gaussian Bayes, AdaBoost (standard and custom), and Random Forest.

Each classifier reports Accuracy, Sensitivity, Specificity, Precision, F1 score,
and ROC-AUC, on both the training and test sets.

Interactive Plotly figures (violin plots, ROC curves, correlation matrix, PCA and
LDA projections) are written as standalone HTML files to `src/plots/` when the
pipeline runs. That directory is regenerated on each run and is not tracked in
version control.

## Project structure

```
.
├── README.md
├── requirements.txt
├── docs/
│   ├── AC_Report1.pdf                       # written report
│   ├── Project_assignment_ML_2025_2026.pdf  # assignment brief
│   └── report/                              # LaTeX sources for the report
└── src/
    ├── main.py                              # entry point and pipeline orchestration
    ├── data/                                # dataR2 dataset (csv and xlsx)
    ├── preprocessing/
    │   └── data_loader.py                   # load, standardize, split
    ├── feature_selection/
    │   ├── roc_auc.py
    │   └── kruskal_wallis.py
    ├── dimensionality_reduction/
    │   ├── pca.py
    │   └── lda.py
    ├── classifiers/
    │   ├── run_all.py                       # runs every classifier / feature set
    │   ├── euclidean.py
    │   ├── mahalanobis.py
    │   ├── fisher.py
    │   ├── svm.py
    │   ├── knn.py
    │   ├── decision_tree.py
    │   ├── bayes.py
    │   ├── adaboost.py
    │   └── random_forest.py
    ├── tuning/
    │   ├── grid_search.py                   # per-model grid search routines
    │   └── run_grid_search.py               # k-fold hyperparameter search driver
    └── plots/                               # generated HTML figures (git-ignored)
```

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

The modules resolve each other relative to the `src/` directory, so run the
pipeline from inside `src/`:

```bash
cd src
python main.py
```

By default `main.py` calls `run_all_classifiers_multiple_times()`, which executes
the full pipeline over 5 randomized splits and prints the mean and standard
deviation of every metric, plus the median feature-importance rankings across runs.

Two other entry points are available at the bottom of `main.py` and can be enabled
by editing the `__main__` block:

- `one_run()`: a single pass through the pipeline
- `run_kfold_hyperparam_multiple_times(n_runs=5)`: repeated k-fold hyperparameter
  search with majority-vote selection of the best parameters

## Notes

- Randomized splits use a fresh seed on every run, so exact numbers vary between
  executions. Reported results are aggregated over multiple runs to account for
  this variance.
- The generated plots and Python bytecode are intentionally excluded from version
  control (see `.gitignore`).

## Authors

- Mariana Duarte
- Miguel Castela
