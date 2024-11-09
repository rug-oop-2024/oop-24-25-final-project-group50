print(mean_squared_error(y_true=y_true, y_pred=y_pred))
print(get_metric("mean_squared_error").metric_function(y_pred, y_true))