"""Download the UCI Online Retail II dataset into data/raw/.

    python -m src.download
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import requests

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
XLSX_NAME = "online_retail_II.xlsx"


def download(force: bool = False) -> Path:
    """Fetch and unzip the dataset; skip if the Excel file is already present."""
    target = RAW_DIR / XLSX_NAME
    if target.exists() and not force:
        print(f"Already downloaded: {target}")
        return target

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "online_retail_ii.zip"
    print(f"Downloading {URL} ...")
    # requests bundles its own CA certificates (python.org builds on macOS don't).
    with requests.get(URL, stream=True, timeout=120) as resp, open(zip_path, "wb") as out:
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=1 << 20):
            out.write(chunk)

    with zipfile.ZipFile(zip_path) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".xlsx"))
        with zf.open(member) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    zip_path.unlink()
    print(f"Saved {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


if __name__ == "__main__":
    download()
