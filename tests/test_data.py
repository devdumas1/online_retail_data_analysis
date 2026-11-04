import numpy as np
import pandas as pd

from src.data import clean


def raw_frame():
    rows = [
        # invoice, stock_code, quantity, date, price, customer_id
        ("100", "85123A", 2, "2011-01-05", 2.5, 1.0),      # kept
        ("100", "85123A", 2, "2011-01-05", 2.5, 1.0),      # exact duplicate -> dropped
        ("101", "22423", 1, "2011-01-06", 10.0, np.nan),   # no customer -> dropped
        ("102", "POST", 1, "2011-01-07", 18.0, 2.0),       # postage -> dropped
        ("103", "gift_0001_10", 1, "2011-01-07", 10.0, 2.0),  # gift voucher -> dropped
        ("104", "DCGS0058", 1, "2011-01-08", 3.0, 2.0),    # irregular code, real product -> kept
        ("105", "21212", 5, "2011-01-09", 0.0, 3.0),       # zero price -> dropped
        ("C106", "85123A", -1, "2011-01-10", 2.5, 1.0),    # cancellation -> kept separately
        ("107", "21212", 3, "2011-12-02", 1.0, 3.0),       # December 2011 -> dropped
    ]
    df = pd.DataFrame(rows, columns=["invoice", "stock_code", "quantity", "invoice_date",
                                     "price", "customer_id"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["description"] = "x"
    df["country"] = "United Kingdom"
    return df


def test_clean_applies_rules():
    data = clean(raw_frame())
    assert sorted(data.purchases["invoice"]) == ["100", "104"]
    assert data.cancellations["invoice"].tolist() == ["C106"]
    assert (data.purchases["amount"] > 0).all()
    assert (data.cancellations["amount"] < 0).all()
    assert data.purchases["customer_id"].dtype == "int64"
    assert data.purchases["invoice_date"].max() < pd.Timestamp("2011-12-01")


def test_report_accounts_for_every_row():
    raw = raw_frame()
    report = clean(raw).report.set_index("step")
    removed = report["removed"].sum()
    kept = report.loc["  -> purchases", "rows"] + report.loc["  -> cancellations", "rows"]
    assert removed + kept == len(raw)
