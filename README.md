# XGBoost Feature Importance Analysis

This repository contains the Python code used for exploratory feature-importance analysis with XGBoost.

The code fits XGBoost regression models for two outcome variables:

- BOP improvement
- PD improvement

Feature importance is evaluated using XGBoost gain importance and SHAP-like contribution values calculated with `pred_contribs=True`.

The mean absolute SHAP-like value is used as the main feature-importance metric.

## Requirements

The code was tested on Windows with Python 3.10.

Required Python packages:

```text
pandas
numpy
xgboost
shap
matplotlib
scikit-learn
openpyxl
```

No special hardware is required.

## Installation

Install the required packages with:

```bash
pip install pandas numpy xgboost shap matplotlib scikit-learn openpyxl
```

Typical installation time is less than 5 minutes on a standard desktop computer.

## Input data

No demo dataset is included.

To run the code, provide an Excel file containing the following columns:

```text
Serum IgG
Serum IgA
Salivary IgG
Salivary sIgA
Pg菌百分比16s
BOP improvement
PD improvement
```

In the code, `Pg菌百分比16s` is renamed as `Pg percentage 16S` for output display.

## How to run

Update the data path in the script:

```python
df = pd.read_excel(r"C:\Users\jp5\Desktop\data")
```

For example, change it to:

```python
df = pd.read_excel("data.xlsx")
```

Then run:

```bash
python xgboost_feature_importance.py
```

## Output

For each outcome variable, the script prints:

- in-sample R2, RMSE, and MAE
- XGBoost gain importance
- SHAP-like feature importance
- overall feature-importance ranking

The script also generates feature-importance bar plots and SHAP summary plots.

Expected run time for a small tabular dataset is usually less than 1 minute.

## Notes

This analysis is exploratory. It is intended to evaluate feature contributions within the current dataset, not to build an externally validated predictive model.

No training/test split is used. Missing feature values are kept as `NaN` and handled directly by XGBoost.

The results should be interpreted as model-derived associations, not as causal effects.
