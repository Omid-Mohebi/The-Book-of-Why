from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from causal_model import CausalDAG


class BackdoorEstimator:

    def __init__(self, dag: CausalDAG) -> None:
        self.dag = dag

    def estimate(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        adjustment_set: set[str] | None = None,
    ) -> dict:

        if adjustment_set is None:
            valid_sets = self.dag.find_backdoor_adjustment_sets(treatment, outcome)
            if not valid_sets:
                raise ValueError(
                    f"No valid backdoor adjustment set found for "
                    f"P({outcome} | do({treatment}))."
                )
            adjustment_set = min(valid_sets, key=len)

        covariates = [treatment] + sorted(adjustment_set)
        X = data[covariates].values
        X = np.column_stack([np.ones(len(X)), X])   # add intercept
        y = data[outcome].values

        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ (X.T @ y)
        y_hat = X @ beta
        residuals = y - y_hat
        n, p = X.shape
        sigma2 = (residuals @ residuals) / (n - p)
        var_beta = sigma2 * XtX_inv

        effect = beta[1]
        se = np.sqrt(var_beta[1, 1])
        t_stat = effect / se
        p_value = 2 * stats.t.sf(abs(t_stat), df=n - p)
        ci_low = effect - 1.96 * se
        ci_high = effect + 1.96 * se

        return {
            "effect": round(effect, 4),
            "std_error": round(se, 4),
            "t_stat": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4),
            "adjustment_set": adjustment_set,
            "method": "backdoor_adjustment",
        }

def naive_regression(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
) -> dict:

    X = np.column_stack([np.ones(len(data)), data[treatment].values])
    y = data[outcome].values
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    y_hat = X @ beta
    residuals = y - y_hat
    n, p = X.shape
    sigma2 = (residuals @ residuals) / (n - p)
    var_beta = sigma2 * XtX_inv

    effect = beta[1]
    se = np.sqrt(var_beta[1, 1])
    t_stat = effect / se
    p_value = 2 * stats.t.sf(abs(t_stat), df=n - p)

    return {
        "effect": round(effect, 4),
        "std_error": round(se, 4),
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "ci_low": round(effect - 1.96 * se, 4),
        "ci_high": round(effect + 1.96 * se, 4),
        "adjustment_set": set(),
        "method": "naive_regression",
    }
