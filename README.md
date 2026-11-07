# Customer Churn Prediction with Engineered Features and Cost-Based Evaluation

Predicting which customers of a UK online gift retailer stop buying, using the
[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) transactions.

**Questions**
1. Do behavior features built from raw transactions predict churn better than simple baselines and raw variables?
2. Which decision threshold maximizes expected profit, given realistic costs of acting on a prediction?
3. What does this evidence show, and what does it *not* show (prediction vs. causation)?

**Status:** data cleaning and churn labels done (notebook 01); customer measures built
(notebook 03). Benchmarks, models and the profit analysis are next. Headline results will go here.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.download        # data -> data/raw/ (see data/README.md)
pytest                        # unit tests, incl. no-leakage checks on the labels
```

## Churn definition
Snapshot design: at each cutoff date, customers with a purchase in the prior 365 days are
scored using only transactions before the cutoff; churn = no purchase in the next N days
(default 90; 60 and 120 as sensitivity checks). Training uses earlier cutoffs whose label
windows close before the first test cutoff.

## Layout
```
src/           download, cleaning (data.py), labels (labels.py), measures (features.py)
tests/         unit tests
notebooks/     01_clean_and_define_churn.ipynb, ...
reports/       figures/ and the write-up
data/          gitignored; see data/README.md
```
