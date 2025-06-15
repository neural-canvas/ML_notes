https://scikit-learn.org/stable/api/sklearn.metrics.html
## Regression
mae, mse, rmse, r2_score

**MAE**
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html

$$MAE = \frac{1}{n}\sum^{n}_{i=1}|{y_{i}-\hat{y{i}}}|$$
scale-dependent accuracy

**MSE**
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html#sklearn.metrics.mean_squared_error
$$MSE = \frac{1}{n}\sum^{n}_{i=1}(Y_{i}-\hat{Y_{i}})^{2}$$



## Classification
### Binary Classification
acurracy, precision, recall, roc_auc_score,



roc_auc_score - 
The AUC-ROC score, ranging from **0 to 1**, provides a single value representing the overall performance of a binary classifier. A score of 0.5 means the classifier performs no better than random guessing, while a score of 1.0 indicates perfect performance.
### Multiclass Classification
log loss
$$\log loss = y_{i}\log(P(y == 1)) + (1-y_{i})\log(1-P(y==1))$$
negative log loss
$$\log loss = - y_{i}\log(P(y == 1)) + (1-y_{i})\log(1-P(y==1))$$



