"""Hidden scoring of a final equation on held-out data. Never called by harnesses.

Splits:  train_dist  new points from the training range (interpolation)
         ood         points beyond the training range (out of distribution: extrapolation)
Metrics: MSE (in the target's units; train and OOD of one task are directly comparable),
         R^2 = 1 - MSE / var(y in this split), and NMSE = MSE / var(y_train).
         Across tasks, MSE is averaged in log space (a geometric mean): target scales differ by
         up to 16 orders of magnitude, so an arithmetic mean would be one task's MSE.
The constants are the ones fitted on training data; they are not refit.
Non-finite predictions count as failure (R^2 = -inf, NMSE = inf).
"""
import numpy as np

from ..benchmark.evaluator import ExpressionError, parse_expression, predict, r2_score

SPLITS = {"train_dist": ("x_train_dist", "y_train_dist"), "ood": ("x_ood", "y_ood")}


def score_equation(task, expression, params, split="ood", cfg=None):
    """R^2 and NMSE of `expression` with fitted `params` on a held-out split."""
    X, y = (getattr(task, a) for a in SPLITS[split])
    if X is None or expression is None:
        return None
    failed = {"r2": float("-inf"), "mse": float("inf"), "nmse": float("inf"), "valid": False}
    try:
        code, _, _ = parse_expression(expression, task.variable_names, max_params=99, max_nodes=10_000)
        y_pred = predict(code, task.variable_names, params, X)
    except ExpressionError:
        return failed
    if not np.all(np.isfinite(y_pred)):
        return failed
    mse = float(np.mean((y_pred - y) ** 2))
    return {"r2": r2_score(y, y_pred), "mse": mse, "nmse": mse / float(np.var(task.y_train)), "valid": True}

