# code-for-manuscript
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
