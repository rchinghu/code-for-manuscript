"""
Exploratory Interpretable Modeling Strategy

This analysis was designed as an exploratory and interpretive modeling
procedure rather than a conventional predictive modeling framework.

The primary objective was not to develop a model with external predictive
generalizability, but to characterize the relationships among variables
within the current dataset and to quantify the relative contribution of
each feature to the fitted model output.

Accordingly, no training/test split was performed. All available samples
were used to train the XGBoost regression model, allowing the model to
fit the current dataset as fully as possible and to capture potential
nonlinear associations and feature interactions.

Missing feature values were not imputed by mean, median, or other
replacement strategies. Instead, original missing values were retained
as NaN and handled natively by XGBoost, in order to avoid introducing
additional assumptions through imputation.

Feature contributions were estimated using XGBoost's native
pred_contribs=True function, which provides SHAP-like contribution
values for each feature. The mean absolute SHAP value was used as the
primary metric for ranking feature importance.

The resulting feature importance should therefore be interpreted as
model-derived associations within the current dataset. These findings
should not be interpreted as causal effects or as externally validated
predictive factors.
"""


import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import matplotlib

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

matplotlib.rcParams['font.sans-serif'] = ['Arial', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_excel(r"C:\Users\jp5\Desktop\data")

print("Detected columns:")
print(df.columns.tolist())

df = df.rename(columns={
    'Pg菌百分比16s': 'Pg percentage 16S'
})


feature_cols = [
    'Serum IgG',
    'Serum IgA',
    'Salivary IgG',
    'Salivary sIgA',
    'Pg percentage 16S'
]


target_cols = [
    "BOP improvement",
    "PD improvement"
]


params = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "eta": 0.03,
    "max_depth": 4,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "lambda": 0,
    "alpha": 0,
    "seed": 42
}


all_results = {}


for target_col in target_cols:

    print("\n" + "=" * 60)
    print(f"Analyzing outcome variable: {target_col}")
    print("=" * 60)

    required_cols = feature_cols + [target_col]

    data = df[required_cols].copy()

    for col in required_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=[target_col])

    X = data[feature_cols]
    y = data[target_col]

    print("Number of valid samples:", len(data))
    print("\nMissing values in each column:")
    print(data.isna().sum())


    dtrain = xgb.DMatrix(
        X,
        label=y,
        missing=np.nan,
        feature_names=feature_cols
    )


    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000
    )


    y_pred = model.predict(dtrain)

    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)

    print("\nIn-sample model fit:")
    print("R2:", r2)
    print("RMSE:", rmse)
    print("MAE:", mae)


    importance = model.get_score(importance_type="gain")

    gain_importance = pd.DataFrame({
        "Feature": feature_cols,
        "Gain": [importance.get(f, 0) for f in feature_cols]
    }).sort_values(by="Gain", ascending=False)

    print("\nXGBoost gain importance:")
    print(gain_importance)

    shap_values_full = model.predict(
        dtrain,
        pred_contribs=True
    )

    shap_values = shap_values_full[:, :-1]

    shap_importance = pd.DataFrame({
        "Feature": feature_cols,
        "Mean_abs_SHAP": np.abs(shap_values).mean(axis=0)
    }).sort_values(by="Mean_abs_SHAP", ascending=False)

    print("\nSHAP feature importance:")
    print(shap_importance)

    summary = shap_importance.merge(
        gain_importance,
        on="Feature",
        how="left"
    )

    summary["SHAP_rank"] = summary["Mean_abs_SHAP"].rank(
        ascending=False,
        method="min"
    )

    summary["Gain_rank"] = summary["Gain"].rank(
        ascending=False,
        method="min"
    )

    summary = summary.sort_values("SHAP_rank")

    print("\nOverall feature importance ranking:")
    print(summary)

    all_results[target_col] = {
        "model": model,
        "fit": {
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae
        },
        "gain_importance": gain_importance,
        "shap_importance": shap_importance,
        "summary": summary,
        "shap_values": shap_values,
        "X": X,
        "y": y
    }

    plt.figure(figsize=(8, 5))

    plot_gain = gain_importance.sort_values("Gain", ascending=True)

    plt.barh(plot_gain["Feature"], plot_gain["Gain"])
    plt.xlabel("Importance: Gain")
    plt.title(f"XGBoost Gain Importance: {target_col}")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))

    plot_shap = shap_importance.sort_values("Mean_abs_SHAP", ascending=True)

    plt.barh(plot_shap["Feature"], plot_shap["Mean_abs_SHAP"])
    plt.xlabel("Mean absolute SHAP value")
    plt.title(f"SHAP Feature Importance: {target_col}")
    plt.tight_layout()
    plt.show()

    shap.summary_plot(
        shap_values,
        X,
        plot_type="dot",
        feature_names=feature_cols
    )

print("\n" + "=" * 60)
print("Final results summary")
print("=" * 60)

for target_col in target_cols:
    print(f"\nOutcome variable: {target_col}")
    print(all_results[target_col]["summary"])
