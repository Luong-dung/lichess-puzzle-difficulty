import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, prediction):
    return {"mae": mean_absolute_error(y_true, prediction),
            "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
            "r2": r2_score(y_true, prediction)}
