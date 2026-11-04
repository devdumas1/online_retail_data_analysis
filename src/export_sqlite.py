"""Export the data to a SQLite database for browsing with SQL tools (e.g. DB Browser for SQLite).

    python -m src.export_sqlite

Creates data/processed/retail.db (gitignored) with three tables:
- raw            every row of the Excel file, before cleaning
- purchases      cleaned purchase lines (see src/data.py)
- cancellations  cleaned cancellation lines
Dates are stored as ISO text ("2010-12-01 08:26:00"), which sorts correctly in SQL.
"""

from __future__ import annotations

import sqlite3

from src.data import PROCESSED_DIR, load_clean, load_raw

DB_PATH = PROCESSED_DIR / "retail.db"


def export(path=DB_PATH) -> None:
    data = load_clean()
    raw = load_raw()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)

    with sqlite3.connect(path) as con:
        for name, df in [("raw", raw), ("purchases", data.purchases),
                         ("cancellations", data.cancellations)]:
            df = df.assign(invoice_date=df["invoice_date"].dt.strftime("%Y-%m-%d %H:%M:%S"))
            df.to_sql(name, con, index=False, chunksize=50_000)
            con.execute(f"CREATE INDEX idx_{name}_customer ON {name}(customer_id)")
            con.execute(f"CREATE INDEX idx_{name}_date ON {name}(invoice_date)")
            print(f"{name}: {len(df):,} rows")
    print(f"Saved {path} ({path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    export()
