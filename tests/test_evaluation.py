import numpy as np
import pandas as pd
import pytest

from src.evaluation import (bootstrap_ci, expected_value_contact, fit_predict, make_boosting,
                            make_logistic, precision_at, realized_profit)


def test_precision_at_takes_highest_scores():
    y = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    score = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]
    assert precision_at(y, score, 0.2) == 0.5      # top 2: 1, 0
    assert precision_at(y, score, 0.3) == pytest.approx(2 / 3)


def test_profit_only_churners_can_be_saved():
    churned = [1, 0, 1]
    spend = [1000, 1000, 100]
    # gains: 0.2 * 0.25 * spend for churners -> 50, 0, 5; cost 10 each
    assert realized_profit([1, 1, 1], churned, spend, 10, 0.2, 0.25) == pytest.approx(50 + 0 + 5 - 30)
    assert realized_profit([1, 0, 0], churned, spend, 10, 0.2, 0.25) == pytest.approx(40)
    assert realized_profit([0, 0, 0], churned, spend, 10, 0.2, 0.25) == 0


def test_expected_value_rule():
    contact = expected_value_contact([0.5, 0.5, 0.1], [1000, 100, 1000], 10, 0.2, 0.25)
    assert contact.tolist() == [True, False, False]   # expected gains 25, 2.5, 5 vs cost 10


def test_bootstrap_resamples_whole_customers():
    groups = np.repeat(np.arange(50), 2)
    values = np.repeat(np.arange(50, dtype=float), 2)
    lo, hi = bootstrap_ci(lambda ix: values[ix].mean(), groups, n_boot=200, seed=1)
    assert lo < values.mean() < hi
    assert bootstrap_ci(lambda ix: values[ix].mean(), groups, n_boot=50, seed=1) == \
        bootstrap_ci(lambda ix: values[ix].mean(), groups, n_boot=50, seed=1)


@pytest.mark.parametrize("make_model", [make_logistic, make_boosting])
def test_models_learn_a_simple_signal_and_accept_missing_values(make_model):
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    df = pd.DataFrame({"a": x, "b": np.where(rng.random(2000) < 0.2, np.nan, rng.normal(size=2000)),
                       "churn": (x + rng.normal(scale=0.5, size=2000) > 0).astype(int)})
    p = fit_predict(make_model, ["a", "b"], df.iloc[:1500], df.iloc[1500:])
    assert precision_at(df["churn"].iloc[1500:], p, 0.2) > 0.9
