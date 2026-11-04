import pandas as pd
import pytest

from src.labels import build_snapshots, history_before, split_cutoffs, valid_cutoffs

CUTOFF = pd.Timestamp("2011-03-01")
END = pd.Timestamp("2011-12-01")


def tx(rows):
    """rows: list of (customer_id, date) -> minimal purchases frame."""
    df = pd.DataFrame(rows, columns=["customer_id", "invoice_date"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"], format="ISO8601")
    df["amount"] = 10.0
    return df


def label_of(snap, customer):
    return snap.set_index("customer_id").loc[customer, "churn"]


@pytest.fixture
def purchases():
    return tx([
        (1, "2011-01-15"), (1, "2011-04-01"),   # active, returns within 90 days -> 0
        (2, "2011-02-10"),                       # active, never returns -> 1
        (3, "2010-12-01"), (3, "2011-07-01"),   # active, returns only after window -> 1
        (4, "2011-03-15"),                       # first purchase after cutoff -> excluded
        (5, "2009-12-15"), (5, "2011-04-01"),   # last purchase before lookback -> excluded
    ])


def test_labels_and_population(purchases):
    snap = build_snapshots(purchases, [CUTOFF], horizon_days=90, active_days=365, data_end=END)
    assert sorted(snap["customer_id"]) == [1, 2, 3]
    assert label_of(snap, 1) == 0
    assert label_of(snap, 2) == 1
    assert label_of(snap, 3) == 1


def test_window_boundaries():
    p = tx([
        (1, "2011-02-01"), (1, "2011-03-01 00:00"),   # exactly at cutoff: counts as a return
        (2, "2011-02-01"), (2, "2011-05-30 00:00"),   # exactly at cutoff + 90d: outside window
        (3, "2010-03-01 00:00"),                      # exactly cutoff - 365d: inside lookback
        (4, "2011-02-28 23:59"),                      # just before cutoff: active
    ])
    snap = build_snapshots(p, [CUTOFF], horizon_days=90, active_days=365, data_end=END)
    assert label_of(snap, 1) == 0
    assert label_of(snap, 2) == 1
    assert set(snap["customer_id"]) == {1, 2, 3, 4}
    assert history_before(p, CUTOFF)["invoice_date"].max() < CUTOFF


def test_history_never_contains_cutoff_or_later(purchases):
    hist = history_before(purchases, CUTOFF)
    assert (hist["invoice_date"] < CUTOFF).all()
    later = purchases[purchases["invoice_date"] >= CUTOFF]
    assert hist.index.intersection(later.index).empty


def test_population_and_history_ignore_future_data(purchases):
    """Changing anything on/after the cutoff must not change who is scored or what
    features can see; only labels may change."""
    future_changed = pd.concat([
        purchases[purchases["invoice_date"] < CUTOFF],
        tx([(2, "2011-03-05"), (99, "2011-03-02"), (1, "2011-11-01")]),
    ], ignore_index=True)
    a = build_snapshots(purchases, [CUTOFF], data_end=END)
    b = build_snapshots(future_changed, [CUTOFF], data_end=END)
    assert a["customer_id"].tolist() == b["customer_id"].tolist()
    pd.testing.assert_frame_equal(
        history_before(purchases, CUTOFF).reset_index(drop=True),
        history_before(future_changed, CUTOFF).reset_index(drop=True),
    )
    assert label_of(b, 2) == 0  # the new 2011-03-05 purchase is (correctly) seen by the label


def test_labels_ignore_data_after_label_window(purchases):
    after = tx([(2, "2011-06-15"), (2, "2011-10-01")])  # after cutoff + 90 days
    a = build_snapshots(purchases, [CUTOFF], data_end=END)
    b = build_snapshots(pd.concat([purchases, after]), [CUTOFF], data_end=END)
    pd.testing.assert_frame_equal(a, b)


def test_rejects_label_window_past_data_end(purchases):
    with pytest.raises(ValueError, match="after the data ends"):
        build_snapshots(purchases, ["2011-09-15"], horizon_days=90, data_end=END)


def test_multiple_cutoffs_are_independent(purchases):
    cutoffs = [pd.Timestamp("2011-02-01"), CUTOFF]
    both = build_snapshots(purchases, cutoffs, data_end=END)
    for c in cutoffs:
        single = build_snapshots(purchases, [c], data_end=END)
        pd.testing.assert_frame_equal(
            both[both["cutoff"] == c].reset_index(drop=True), single
        )


def test_valid_cutoffs_have_full_history_and_label_window():
    for horizon in (60, 90, 120):
        cuts = valid_cutoffs(horizon, min_history_days=365)
        assert cuts, horizon
        assert cuts[0] >= pd.Timestamp("2009-12-01") + pd.Timedelta(days=365)
        assert cuts[-1] + pd.Timedelta(days=horizon) <= END


def test_split_train_labels_end_before_test():
    cuts = valid_cutoffs(90)
    train, test = split_cutoffs(cuts, horizon_days=90, n_test=3)
    assert test == cuts[-3:]
    assert all(c + pd.Timedelta(days=90) <= test[0] for c in train)
    assert max(train) < min(test)
