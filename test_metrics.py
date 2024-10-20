from autoop.core.ml.metric import get_metric
from sklearn.metrics import cohen_kappa_score, mean_squared_error, accuracy_score, log_loss, mean_absolute_percentage_error, r2_score
import numpy as np

METRICS = [
    "mean_squared_error",
    "accuracy",
    "log_loss",
    "mean_absolute_percentage_error",
    "cohens_kappa",
    "r_squared_score",
]

actual_labels = np.array(['cat', 'dog', 'rabbit', 'cat', 'dog', 'rabbit', 'cat', 'dog', 'rabbit', 'cat', 'dog', 'rabbit', 'dog', 'cat', 'rabbit'])
predicted_labels = np.array(['dog', 'dog', 'rabbit', 'cat', 'dog', 'rabbit', 'cat', 'dog', 'rabbit', 'cat', 'dog', 'rabbit', 'dog', 'cat', 'rabbit'])

y_true = np.array([100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800])
y_pred = np.array([110, 140, 210, 240, 290, 360, 390, 460, 490, 530, 610, 640, 720, 730, 810])

# ACCURACY TEST
# print(accuracy_score(y_true=actual_labels, y_pred=predicted_labels))
# print(get_metric("accuracy").metric_function(predicted_labels, actual_labels))

# MSE TEST
# print(mean_squared_error(y_true=y_true, y_pred=y_pred))
# print(get_metric("mean_squared_error").metric_function(y_pred, y_true))

# LOG LOSS TEST
# print(log_loss(y_true=actual_labels, y_pred=predicted_labels))
# print(get_metric("log_loss").metric_function(predicted_labels, actual_labels))

# MAPE TEST
# print(mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred))
# print(get_metric("mean_absolute_percentage_error").metric_function(y_pred, y_true))

# COHEN KAPPA TEST
# print(cohen_kappa_score(predicted_labels, actual_labels))
# print(get_metric("cohens_kappa").metric_function(predicted_labels, actual_labels))

# R^2 SCORE TEST
# print(r2_score(y_true=y_true, y_pred=y_pred))
# print(get_metric("r_squared_score").metric_function(y_pred, y_true))