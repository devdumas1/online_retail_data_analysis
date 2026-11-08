from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "online_retail_II.xlsx"
PROCESSED_DIR = ROOT / "data" / "processed"

# Exclusive end of the analysis window: data stops on 2011-12-09, so December 2011 is partial.
ANALYSIS_END = pd.Timestamp("2011-12-01")

# Stock codes that are not products. Checked against the full dataset; codes such as
# DCGS0058 and SP1002 look irregular but are real products, so they are kept.
NON_PRODUCT_CODES = frozenset({
    "POST", "DOT", "C2", "C3", "M", "D", "S", "B", "ADJUST", "ADJUST2",
    "BANK CHARGES", "PADS", "CRUK", "AMAZONFEE", "TEST001", "TEST002", "GIFT",
})
NON_PRODUCT_PREFIXES = ("GIFT_",)  # gift vouchers, e.g. gift_0001_10

COLUMNS = {
    "Invoice": "invoice", "StockCode": "stock_code", "Description": "description",
    "Quantity": "quantity", "InvoiceDate": "invoice_date", "Price": "price",
    "Customer ID": "customer_id", "Country": "country",
}


@dataclass
class CleanData:
    purchases: pd.DataFrame      # one row per purchased line item; amount > 0
    cancellations: pd.DataFrame  # cancelled line items; quantity and amount < 0
    report: pd.DataFrame         # rows remaining / removed at each cleaning step


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read both yearly sheets into one frame with snake_case columns."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.download` first.")
    sheets = pd.read_excel(
        path, sheet_name=None,
        dtype={"Invoice": str, "StockCode": str, "Description": str, "Country": str},
    )
    return pd.concat(sheets.values(), ignore_index=True).rename(columns=COLUMNS)


def is_non_product(codes: pd.Series) -> pd.Series:
    upper = codes.str.strip().str.upper()
    return upper.isin(NON_PRODUCT_CODES) | upper.str.startswith(NON_PRODUCT_PREFIXES)


def clean(raw: pd.DataFrame, end: pd.Timestamp = ANALYSIS_END) -> CleanData:
    """Apply the cleaning rules to a frame produced by `load_raw`."""
    steps = [("raw rows", len(raw))]
    df = raw.copy()

    df = df[df["customer_id"].notna()]
    steps.append(("drop missing customer ID", len(df)))

    df = df.drop_duplicates()
    steps.append(("drop exact duplicates", len(df)))

    df = df[df["invoice_date"] < end]
    steps.append((f"drop on/after {end.date()}", len(df)))

    df = df[~is_non_product(df["stock_code"])]
    steps.append(("drop non-product codes", len(df)))

    df = df[df["price"] > 0]
    steps.append(("drop price <= 0", len(df)))

    df = df.assign(
        customer_id=df["customer_id"].astype("int64"),
        stock_code=df["stock_code"].str.strip(),
        amount=df["quantity"] * df["price"],
    )
    is_cancel = df["invoice"].str.upper().str.startswith("C")
    purchases = df[~is_cancel].sort_values("invoice_date").reset_index(drop=True)
    cancellations = df[is_cancel].sort_values("invoice_date").reset_index(drop=True)
    if (purchases["quantity"] <= 0).any():
        raise ValueError("Non-cancelled invoice with non-positive quantity; check the source data.")

    report = pd.DataFrame(steps, columns=["step", "rows"])
    report["removed"] = (-report["rows"].diff()).fillna(0).astype(int)
    report = pd.concat([report, pd.DataFrame([
        {"step": "  -> purchases", "rows": len(purchases), "removed": 0},
        {"step": "  -> cancellations", "rows": len(cancellations), "removed": 0},
    ])], ignore_index=True)
    return CleanData(purchases, cancellations, report)


def load_clean(use_cache: bool = True) -> CleanData:
    """Cleaned data, read from data/processed/ if cached (the Excel load takes ~1 min)."""
    paths = {name: PROCESSED_DIR / f"{name}.parquet"
             for name in ("purchases", "cancellations", "cleaning_report")}
    if use_cache and all(p.exists() for p in paths.values()):
        return CleanData(*(pd.read_parquet(p) for p in paths.values()))

    data = clean(load_raw())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data.purchases.to_parquet(paths["purchases"], index=False)
    data.cancellations.to_parquet(paths["cancellations"], index=False)
    data.report.to_parquet(paths["cleaning_report"], index=False)
    return data
