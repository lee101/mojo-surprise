from __future__ import annotations

import numpy as np


def _errors(predictions):
    return np.asarray([prediction.r_ui - prediction.est for prediction in predictions])


def rmse(predictions, verbose=True):
    value = float(np.sqrt(np.mean(_errors(predictions) ** 2)))
    if verbose:
        print(f"RMSE: {value:1.4f}")
    return value


def mse(predictions, verbose=True):
    value = float(np.mean(_errors(predictions) ** 2))
    if verbose:
        print(f"MSE: {value:1.4f}")
    return value


def mae(predictions, verbose=True):
    value = float(np.mean(np.abs(_errors(predictions))))
    if verbose:
        print(f"MAE:  {value:1.4f}")
    return value

