from __future__ import annotations

import numpy as np
import pandas as pd

from src.labels import history_before

WINDOWS = (30, 90, 180, 365)


def orders_before(purchases: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Collapse purchase lines before `cutoff` into one row per order (invoice)."""
    return (history_before(purchases, cutoff)
            .groupby(["customer_id", "invoice"])
            .agg(date=("invoice_date", "min"), value=("amount", "sum"),
                 n_products=("stock_code", "nunique"))
            .reset_index())


def features_at(purchases: pd.DataFrame, cancellations: pd.DataFrame, cutoff,
                customer_ids) -> pd.DataFrame:
    """Measures for `customer_ids` as of `cutoff`, indexed by customer_id."""
    cutoff = pd.Timestamp(cutoff)
    orders = orders_before(purchases, cutoff)
    orders = orders[orders["customer_id"].isin(customer_ids)]
    days_ago = (cutoff - orders["date"]).dt.days
    by_customer = orders.groupby("customer_id")

    f = pd.DataFrame(index=pd.Index(sorted(customer_ids), name="customer_id"))
    f["recency_days"] = (cutoff - by_customer["date"].max()).dt.days
    f["tenure_days"] = (cutoff - by_customer["date"].min()).dt.days

    for w in WINDOWS:
        recent = orders[days_ago < w].groupby("customer_id")
        f[f"orders_{w}d"] = recent.size()
        f[f"spend_{w}d"] = recent["value"].sum()
    counts = [f"{m}_{w}d" for m in ("orders", "spend") for w in WINDOWS]
    f[counts] = f[counts].fillna(0)
    f["log_spend_365d"] = np.log1p(f["spend_365d"])

    previous = orders[(days_ago >= 90) & (days_ago < 180)].groupby("customer_id")["value"].sum()
    previous = previous.reindex(f.index, fill_value=0)
    total = f["spend_90d"] + previous
    f["spend_trend"] = ((f["spend_90d"] - previous) / total.where(total > 0)).fillna(0)

    # Gaps between distinct order dates: several invoices on one day count as one visit.
    days = (orders.assign(day=orders["date"].dt.normalize())
            .drop_duplicates(["customer_id", "day"])
            .sort_values(["customer_id", "day"]))
    gaps = days.groupby("customer_id")["day"].diff().dt.days
    gap_stats = gaps.groupby(days["customer_id"]).agg(["median", "std"])
    f["median_gap_days"] = gap_stats["median"]
    f["gap_std_days"] = gap_stats["std"]
    f["overdue_ratio"] = f["recency_days"] / f["median_gap_days"]
    f["one_time_buyer"] = (days.groupby("customer_id").size() == 1).astype(int)

    last_year = orders[days_ago < 365].groupby("customer_id")
    f["avg_order_value"] = last_year["value"].mean()
    f["products_per_order"] = last_year["n_products"].mean()

    cancelled = history_before(cancellations, cutoff)
    cancelled = cancelled[cancelled["invoice_date"] >= cutoff - pd.Timedelta(days=365)]
    cancelled_value = -cancelled.groupby("customer_id")["amount"].sum()
    f["return_rate"] = (cancelled_value.reindex(f.index, fill_value=0)
                        / f["spend_365d"].where(f["spend_365d"] > 0)).fillna(0)

    autumn = orders.loc[(days_ago < 365) & orders["date"].dt.month.isin([9, 10, 11]), "customer_id"]
    f["autumn_buyer"] = f.index.isin(autumn).astype(int)

    lines = history_before(purchases, cutoff)
    uk_share = (lines["country"] == "United Kingdom").groupby(lines["customer_id"]).mean()
    f["uk"] = (uk_share.reindex(f.index) > 0.5).astype(int)
    return f


def build_features(purchases: pd.DataFrame, cancellations: pd.DataFrame,
                   snapshots: pd.DataFrame) -> pd.DataFrame:
    """Add the measures to every (customer, cutoff) row of `build_snapshots` output."""
    parts = []
    for cutoff, snap in snapshots.groupby("cutoff"):
        feats = features_at(purchases, cancellations, cutoff, snap["customer_id"].unique())
        parts.append(snap.merge(feats, left_on="customer_id", right_index=True, how="left"))
    return pd.concat(parts, ignore_index=True)
