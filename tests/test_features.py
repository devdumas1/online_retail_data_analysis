import numpy as np
import pandas as pd
import pytest

from src.features import build_features, features_at
from src.labels import build_snapshots

CUTOFF = pd.Timestamp("2011-03-01")


def lines(rows):
    """rows: list of (customer_id, invoice, date, stock_code, amount, country)."""
    df = pd.DataFrame(rows, columns=["customer_id", "invoice", "invoice_date", "stock_code",
                                     "amount", "country"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


@pytest.fixture
def purchases():
    return lines([
        (1, "A", "2010-12-01", "X", 10.0, "United Kingdom"),
        (1, "A", "2010-12-01", "Y", 10.0, "United Kingdom"),
        (1, "B", "2011-02-15", "X", 30.0, "United Kingdom"),
        (1, "C", "2011-02-15", "Z", 10.0, "United Kingdom"),   # same day as B: one visit
        (1, "D", "2011-04-01", "X", 99.0, "United Kingdom"),   # after cutoff: ignored
        (2, "E", "2010-10-20", "X", 50.0, "France"),
    ])


@pytest.fixture
def cancellations():
    return lines([
        (1, "C1", "2011-01-10", "X", -6.0, "United Kingdom"),
        (1, "C2", "2011-03-05", "X", -50.0, "United Kingdom"),  # after cutoff: ignored
    ])


def test_measures_match_hand_calculation(purchases, cancellations):
    f = features_at(purchases, cancellations, CUTOFF, [1, 2])
    one, two = f.loc[1], f.loc[2]

    assert one["recency_days"] == 14 and one["tenure_days"] == 90
    assert (one["orders_30d"], one["spend_30d"]) == (2, 40)
    assert (one["orders_90d"], one["spend_90d"]) == (2, 40)      # order A is exactly 90 days ago
    assert (one["orders_365d"], one["spend_365d"]) == (3, 60)
    assert one["spend_trend"] == pytest.approx((40 - 20) / (40 + 20))
    assert one["median_gap_days"] == 76 and one["overdue_ratio"] == pytest.approx(14 / 76)
    assert one["one_time_buyer"] == 0
    assert one["avg_order_value"] == pytest.approx(20)
    assert one["products_per_order"] == pytest.approx(4 / 3)
    assert one["return_rate"] == pytest.approx(6 / 60)
    assert (one["autumn_buyer"], one["uk"]) == (0, 1)

    assert two["recency_days"] == 132 and two["spend_trend"] == -1
    assert two["one_time_buyer"] == 1 and np.isnan(two["overdue_ratio"])
    assert (two["autumn_buyer"], two["uk"], two["return_rate"]) == (1, 0, 0)


def test_future_data_cannot_change_measures(purchases, cancellations):
    before = features_at(purchases, cancellations, CUTOFF, [1, 2])
    future = lines([(c, f"F{c}", "2011-03-01", "X", 1000.0, "France") for c in (1, 2)])
    after = features_at(pd.concat([purchases, future]), pd.concat([cancellations, future]),
                        CUTOFF, [1, 2])
    pd.testing.assert_frame_equal(before, after)


def test_one_row_per_snapshot(purchases, cancellations):
    snaps = build_snapshots(purchases, [CUTOFF], horizon_days=90, active_days=365,
                            data_end=pd.Timestamp("2011-12-01"))
    feats = build_features(purchases, cancellations, snaps)
    assert len(feats) == len(snaps)
    assert feats[["customer_id", "cutoff", "churn"]].equals(snaps[["customer_id", "cutoff", "churn"]])
