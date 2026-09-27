# CodeAlpha_DiseasePrediction

Machine Learning Internship — Task 4: Disease Prediction from Medical Data
(CodeAlpha)

## Overview
This project builds a classification model to predict the presence of heart
disease in patients based on structured medical data, using Logistic
Regression.

## Project Structure
```
CodeAlpha_DiseasePrediction
├── notebooks_or_scripts
│   ├── evaluate_model.py
│   ├── explore_data.py
│   └── train_model.py
├── results
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   └── feature_distributions.png
├── heart.csv
├── requirements.txt
└── README.md
```

## Dataset
- Source: Heart Disease dataset (`heart.csv`)
- 1,025 patient records, 14 columns (13 features + `target`)
- Features include: age, sex, chest pain type (`cp`), resting blood pressure
  (`trestbps`), cholesterol (`chol`), fasting blood sugar (`fbs`), resting ECG
  results (`restecg`), max heart rate achieved (`thalach`), exercise-induced
  angina (`exang`), ST depression (`oldpeak`), slope, number of major vessels
  (`ca`), and thalassemia (`thal`)
- Target: binary — `1` = heart disease present, `0` = no heart disease
- No missing values in any column
- Fairly balanced classes: 526 patients with disease, 499 without

## Approach
Classification techniques, as suggested by the task:

1. **Data exploration**: verified dataset shape, checked for missing values,
   confirmed class balance, and visualized feature correlations and
   distributions split by diagnosis.
2. **Preprocessing**: features and target separated, an 80/20 train-test
   split applied, and features standardized with `StandardScaler` (important
   for Logistic Regression, which is sensitive to feature scale).
3. **Modeling**: **Logistic Regression**, one of the algorithms explicitly
   suggested by the task for credit/medical-style binary classification
   problems.
4. **Evaluation**: accuracy, precision, recall, F1-score, ROC-AUC, full
   classification report, and a confusion matrix — matching the metrics the
   task specifically calls for.

## Results

Evaluated on a held-out test set of 205 patients (20% split):

| Metric | Value |
|---|---|
| **Accuracy** | **79.51%** |
| **Precision** | **75.63%** |
| **Recall** | **87.38%** |
| **F1-Score** | **81.08%** |
| **ROC-AUC** | **87.87%** |

Per-class performance:

```
              precision    recall  f1-score   support

  No Disease       0.85      0.72      0.78       102
     Disease       0.76      0.87      0.81       103

    accuracy                           0.80       205
   macro avg       0.80      0.79      0.79       205
weighted avg       0.80      0.80      0.79       205
```

The model favors **recall on the "Disease" class (87%)** over precision
(76%), meaning it catches most true disease cases but produces some false
positives. In a medical screening context, this trade-off is generally
preferable to the reverse, since missing an actual disease case is typically
costlier than flagging a healthy patient for further testing. The strong
ROC-AUC (87.87%) indicates the model separates the two classes well overall.

Visualizations saved in `results/`:
- `correlation_heatmap.png` — feature correlation matrix
- `feature_distributions.png` — age, cholesterol, resting BP, and max heart
  rate distributions split by diagnosis
- `confusion_matrix.png` — predicted vs actual outcomes on the test set

## How to Run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. From inside `notebooks_or_scripts/`, run in order:
   ```
   python explore_data.py     # explores data, saves correlation_heatmap.png + feature_distributions.png
   python train_model.py      # trains the model, prints test accuracy
   python evaluate_model.py   # full metrics + confusion matrix
   ```

## Tech Stack
- Python
- pandas
- scikit-learn (Logistic Regression, preprocessing, metrics)
- Matplotlib / Seaborn (visualizations)

## Notes
- Logistic Regression was chosen as a strong, interpretable baseline for this
  task, as recommended in the task description. Other suggested algorithms
  (Decision Trees, Random Forest, SVM, XGBoost) could be explored as further
  improvements, along with additional feature engineering.