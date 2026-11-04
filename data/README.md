# Data

**Source:** [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
(Chen, 2019), licensed CC BY 4.0. About 1.07M transaction lines from a UK online gift
retailer, 1 Dec 2009 – 9 Dec 2011, in one Excel file with a sheet per year.

Raw and processed data are **not** committed to git (`data/raw/` and `data/processed/`
are gitignored).

## Download
From the project root, with the virtual environment active:
```bash
python -m src.download
```
This fetches `online+retail+ii.zip` (~45 MB) from UCI and saves
`data/raw/online_retail_II.xlsx`. It does nothing if the file already exists.

Manual alternative: download the zip from the link above, unzip it, and place
`online_retail_II.xlsx` in `data/raw/`.

## Processed files
The first call to `src.data.load_clean()` reads the Excel file (about a minute), applies
the cleaning rules and caches the result in `data/processed/`:

| File | Contents |
|---|---|
| `purchases.parquet` | Cleaned purchase line items (`amount = quantity × price > 0`) |
| `cancellations.parquet` | Cancelled line items (invoice starts with "C"; negative quantity) |
| `cleaning_report.parquet` | Rows remaining/removed at each cleaning step |

Delete `data/processed/` (or call `load_clean(use_cache=False)`) to rebuild.

## Columns
`invoice`, `stock_code`, `description`, `quantity`, `invoice_date`, `price`,
`customer_id`, `country`, `amount`
