# Diabetes Risk Prediction: Missing Data Handling and Logistic Regression

Individual contribution to a 6-member group project for **IT2011 – Artificial Intelligence and Machine Learning** (BSc (Hons) in IT, SLIIT, Year 2 Semester 1, 2026).

The goal is a binary classifier that predicts whether a patient has diabetes (`Outcome = 1`) from 8 routine clinical measurements. This repository contains my two parts of the project:

1. **Missing Data Handling**: preprocessing technique (`notebooks/IT_25103555_MissingData.ipynb`)
2. **Logistic Regression**: classification model (`notebooks/IT_25103555_LogisticRegression.ipynb`)

The other members worked on outlier removal, encoding, scaling, feature selection, PCA, SVM, Random Forest, K-Means and an MLP. Those notebooks are not included here.

## Dataset

- **Pima Indians Diabetes Database**: 768 female patients aged 21+, 500 non-diabetic and 268 diabetic.
- Source: UCI Machine Learning Repository / National Institute of Diabetes and Digestive and Kidney Diseases (Smith et al., 1988).
- A copy is in `data/raw/diabetes.csv`. The notebooks can also download it from a public mirror.
- Features: `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age`.

## What I did

### 1. Missing data handling
The dataset has no NaN values, but five columns use `0` as a placeholder for "not measured", which is clinically impossible for a real patient.

| Column | Zeros (treated as missing) |
|---|---|
| Glucose | 5 (0.7%) |
| BloodPressure | 35 (4.6%) |
| SkinThickness | 227 (29.6%) |
| Insulin | 374 (48.7%) |
| BMI | 11 (1.4%) |

I replaced these zeros with `NaN` and used **median imputation**, which is robust to the skewed Insulin and SkinThickness distributions and keeps all 768 rows. A before-and-after histogram of Insulin is saved in `results/eda_visualizations/`.

### 2. Logistic Regression
- Split the data first (80/20, stratified, `random_state = 42`), and fitted every preprocessing step on the **training data only** to avoid data leakage.
- Compared four hand-picked varieties (regularisation strength `C` and balanced class weights), then tuned with `GridSearchCV` (5-fold, scored on F1 for the diabetic class).
- Re-tested the tuned model under six preprocessing varieties using 5-fold stratified cross-validation, built from the group's preprocessing techniques.

## Results

Held-out test set (154 patients), best preprocessing (StandardScaler plus engineered BMI-category and age-group features):

| Metric | Value |
|---|---|
| Accuracy | 0.753 |
| Recall (diabetic class) | 0.778 |
| F1 (diabetic class) | 0.689 |
| ROC-AUC | 0.826 |
| 5-fold CV F1 (training data) | 0.697 |

- Balanced class weights mattered more than regularisation strength (F1 0.546 → 0.650).
- Scaling plus encoding raised the cross-validated F1 from 0.665 to 0.697.
- Across the group's five classifiers, this model had the highest held-out test F1. Cross-validated F1 scores were all within 0.688–0.699, which is smaller than the fold-to-fold variation, so the models are effectively tied. The conclusion in the group report treats Logistic Regression as the preferred model mainly because it is the easiest to explain.

## Limitations

- Small dataset (768 rows) and a small test set, so single-split scores are noisy.
- Median imputation of almost half of the Insulin values is a strong simplification. KNN or model-based imputation could be tried.
- The population is Pima Indian women aged 21+, so the model should not be assumed to work for other groups. It is a learning project, **not** a diagnostic tool.

## How to run

1. Open the notebooks in Google Colab, or in Jupyter with Python 3.10+.
2. Install the libraries: `pip install pandas numpy scikit-learn matplotlib seaborn`.
3. Run all cells. The notebooks use `data/raw/diabetes.csv` if present and otherwise download the dataset.

## Tools

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Google Colab.

## AI tool usage

The group used Claude (Anthropic) as a support tool for code review, for suggesting the cross-validated preprocessing study, and for drafting documentation, as declared in the group report. All experiments were run by the group.
