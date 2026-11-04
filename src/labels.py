"""Snapshot ("cutoff date") churn labels.

For each cutoff date:
- the population is customers with at least one purchase in the `active_days`
  before the cutoff: [cutoff - active_days, cutoff)
- features may use only transactions strictly before the cutoff (`history_before`)
- churn = 1 if the customer makes no purchase in [cutoff, cutoff + horizon_days)

Time conventions: cutoffs are midnight timestamps; a purchase at exactly the cutoff
belongs to the label window, never to the history.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.data import ANALYSIS_END

DATA_START = pd.Timestamp("2009-12-01")


def history_before(transactions: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Transactions strictly before `cutoff`, the only rows features may use."""
    return transactions[transactions["invoice_date"] < pd.Timestamp(cutoff)]


def build_snapshots(
    purchases: pd.DataFrame,
    cutoffs: Iterable,
    horizon_days: int = 90,
    active_days: int = 365,
    data_end: pd.Timestamp = ANALYSIS_END,
) -> pd.DataFrame:
    """One row per (customer, cutoff) with the churn label.

    Columns: customer_id, cutoff, horizon_days, churn (1 = no purchase in the horizon).
    """
    rows = []
    for cutoff in cutoffs:
        cutoff = pd.Timestamp(cutoff)
        label_end = cutoff + pd.Timedelta(days=horizon_days)
        if label_end > data_end:
            raise ValueError(
                f"Label window for cutoff {cutoff.date()} ends {label_end.date()}, "
                f"after the data ends ({data_end.date()})."
            )
        window_start = cutoff - pd.Timedelta(days=active_days)
        dates = purchases["invoice_date"]
        active = purchases.loc[(dates >= window_start) & (dates < cutoff), "customer_id"].unique()
        returned = purchases.loc[(dates >= cutoff) & (dates < label_end), "customer_id"].unique()

        snap = pd.DataFrame({"customer_id": sorted(active)})
        snap["cutoff"] = cutoff
        snap["horizon_days"] = horizon_days
        snap["churn"] = (~snap["customer_id"].isin(returned)).astype(int)
        rows.append(snap)

    if not rows:
        return pd.DataFrame(columns=["customer_id", "cutoff", "horizon_days", "churn"])
    return pd.concat(rows, ignore_index=True)


def valid_cutoffs(
    horizon_days: int,
    min_history_days: int = 365,
    data_start: pd.Timestamp = DATA_START,
    data_end: pd.Timestamp = ANALYSIS_END,
    freq: str = "MS",
) -> list[pd.Timestamp]:
    """Month-start cutoffs with `min_history_days` of history before them and a
    complete label window after them."""
    first = data_start + pd.Timedelta(days=min_history_days)
    last = data_end - pd.Timedelta(days=horizon_days)
    return list(pd.date_range(first, last, freq=freq))


def split_cutoffs(
    cutoffs: list[pd.Timestamp], horizon_days: int, n_test: int = 3
) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """Use the last `n_test` cutoffs for testing and, for training, only cutoffs whose
    label window closes before the first test cutoff (no overlap with the test period)."""
    cutoffs = sorted(pd.Timestamp(c) for c in cutoffs)
    if n_test >= len(cutoffs):
        raise ValueError("Need more cutoffs than n_test.")
    test = cutoffs[-n_test:]
    train = [c for c in cutoffs[:-n_test] if c + pd.Timedelta(days=horizon_days) <= test[0]]
    if not train:
        raise ValueError("No training cutoff has a label window that ends before the test period.")
    return train, test
