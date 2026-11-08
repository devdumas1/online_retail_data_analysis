# Data

Source: UCI Online Retail II (https://archive.ics.uci.edu/dataset/502/online+retail+ii), CC BY 4.0. About 1.07M sales lines from a UK online gift shop, December 2009 to December 2011.

The data isn't stored on GitHub. To download it into data/raw/:

```bash
python -m src.download
```

The first run of the notebooks cleans the data and saves it to data/processed/. To browse it with SQL, run `python -m src.export_sqlite`, which writes data/processed/retail.db.
