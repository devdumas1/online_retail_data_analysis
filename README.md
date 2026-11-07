# Customer Churn Prediction with Engineered Features and Cost-Based Evaluation

Predicting which customers of a UK online gift retailer stop buying, using the
[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) transactions.

**Questions**
1. Do behavior features built from raw transactions predict churn better than simple baselines and raw variables?
2. Which decision threshold maximizes expected profit, given realistic costs of acting on a prediction?
3. What does this evidence show, and what does it *not* show (prediction vs. causation)?

**Status:** minimum viable version complete (notebooks 01-05, write-up in `reports/report.md`).

**Headline results** (test months July-September 2011)
- Custom behavior measures add only a small, unconfirmed gain over recency, order count and spend
  (PR-AUC 0.794 vs 0.787; 95% CI of the difference -0.002 to +0.016).
- Targeting "please come back" offers by expected value (churn chance x success rate x margin x
  spend > offer cost) was the most profitable policy in all 12 cost scenarios tested: +GBP 12,800
  vs contacting everybody in the main case.
- Profit levels rest on an assumed 20% offer success rate; only a randomized test can measure it.

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
src/           download, cleaning (data.py), labels (labels.py), measures (features.py),
               models, metrics and profit rules (evaluation.py)
tests/         unit tests
notebooks/     01 clean and define churn, 02 steps 1-2 inline, 03 measures,
               04 benchmarks/models/evaluation, 05 profit analysis
reports/       report.md (the write-up) and figures/
data/          gitignored; see data/README.md
```
