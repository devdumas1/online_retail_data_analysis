from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ID_COLUMNS = ("customer_id", "cutoff", "horizon_days", "churn")
RAW_MEASURES = ["recency_days", "orders_365d", "log_spend_365d"]


def measure_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c not in ID_COLUMNS]


def make_logistic():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         StandardScaler(), LogisticRegression(max_iter=2000))


def make_boosting(seed: int = 0):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=100, random_state=seed)


def fit_predict(make_model: Callable, columns: list[str], train: pd.DataFrame,
                test: pd.DataFrame) -> np.ndarray:
    """Fit on the training rows and return churn probabilities for the test rows."""
    model = make_model().fit(train[columns], train["churn"])
    return model.predict_proba(test[columns])[:, 1]


def precision_at(y, score, share: float) -> float:
    """Churn rate among the `share` of customers with the highest scores."""
    y, score = np.asarray(y), np.asarray(score)
    n = max(1, int(round(len(y) * share)))
    return y[np.argsort(-score, kind="stable")[:n]].mean()


def bootstrap_ci(statistic: Callable[[np.ndarray], float], groups, n_boot: int = 1000,
                 seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval for `statistic(row_index)`, resampling whole groups (customers),
    because the same customer appears at several cutoffs."""
    groups = np.asarray(groups)
    codes, uniques = pd.factorize(groups)
    rows_of = np.split(np.argsort(codes, kind="stable"), np.cumsum(np.bincount(codes))[:-1])
    rng = np.random.default_rng(seed)
    stats = [statistic(np.concatenate([rows_of[g] for g in rng.integers(0, len(uniques), len(uniques))]))
             for _ in range(n_boot)]
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(stats, [tail, 100 - tail])
    return float(lo), float(hi)


def offer_gain(churned, spend_365d, success_rate: float, margin: float) -> np.ndarray:
    """Value recovered by contacting each customer: only churners can be saved."""
    return np.asarray(churned) * success_rate * margin * np.asarray(spend_365d)


def realized_profit(contact, churned, spend_365d, offer_cost: float, success_rate: float,
                    margin: float) -> float:
    contact = np.asarray(contact, dtype=bool)
    gain = offer_gain(churned, spend_365d, success_rate, margin)
    return float((gain[contact] - offer_cost).sum())


def expected_value_contact(p_churn, spend_365d, offer_cost: float, success_rate: float,
                           margin: float) -> np.ndarray:
    """Contact when the expected value saved exceeds the cost of the offer."""
    return np.asarray(p_churn) * success_rate * margin * np.asarray(spend_365d) > offer_cost
