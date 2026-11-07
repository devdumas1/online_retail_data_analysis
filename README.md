# Customer Churn Prediction & Profit Analysis

**Python · pandas · scikit-learn · statistical inference**

Predicting which customers of a UK online gift retailer will stop buying, and deciding who should
get a "please come back" offer, using the
[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) transactions
(December 2009 to November 2011).

## Highlights

- **Flagged at-risk customers with 86% precision on unseen months, against 78% for a recency rule.**
  Built a leakage-free churn pipeline on 1.07M transactions (42,850 customer snapshots) with
  time-based validation and 20 automated tests.
- **Increased estimated retention profit by £12.8K per quarter (+7.5%) over contacting every
  customer.** Built an expected-value targeting rule that beat simpler policies in all 12 cost
  scenarios tested.
- **Showed that 21 engineered features did not significantly beat simple recency, frequency and
  spend measures**, using customer-level bootstrap confidence intervals. Designed a randomized
  experiment to measure the offer's true effect.

Full write-up: [`reports/report.md`](reports/report.md).

## Results (test months July–September 2011)

| Method | PR-AUC (95% CI) | Churn rate in top 10% flagged |
|---|---|---|
| No model | 0.562 | 56% |
| Recency rule | 0.738 (0.721–0.753) | 78% |
| Logistic regression, recency/orders/spend | 0.788 (0.773–0.802) | 84% |
| Gradient boosting, all 21 features | 0.794 (0.779–0.808) | 86% |

| Targeting policy (£10 offer, 20% success, 25% margin) | Share contacted | Profit |
|---|---|---|
| Everybody | 100% | £170,900 |
| By spend only | 74% | £180,700 |
| **Expected value, gradient boosting** | **78%** | **£183,700** |

Profit levels rest on the assumed 20% offer success rate; the ranking of policies held for success
rates of 10–30% and offer costs of £5–£50. Only a randomized test can measure the real rate.

## Method

1. **Clean** 1.07M purchase lines to 776,855: drop lines with no customer ID, duplicates, the
   incomplete December 2011, non-product charges and zero prices; keep cancellations separately.
2. **Define churn** at monthly cutoff dates: a customer who bought in the previous 365 days has
   churned if they buy nothing in the next 90 days (60 and 120 days as checks). Models learn from
   December 2010–April 2011 and are judged on July–September 2011, with a gap so the two never overlap.
3. **Build 21 features** per customer per cutoff from purchases before the cutoff only: recency,
   tenure, orders and spend over 30/90/180/365 days, spend trend, buying rhythm (usual gap between
   orders and how overdue the customer is), basket size, return rate, autumn buying and location.
4. **Benchmark and model:** no model, a recency rule and logistic regression on raw measures,
   against logistic regression and gradient boosting on all features.
5. **Evaluate** with PR-AUC and top-k precision, customer-level bootstrap intervals, calibration,
   permutation importance and robustness checks fixed before modelling.
6. **Profit analysis:** contact a customer when churn chance × success rate × margin × annual spend
   exceeds the offer cost, compared with simpler policies across 12 cost scenarios.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.download        # data -> data/raw/ (see data/README.md)
pytest                        # 20 tests, incl. checks that no future data leaks into labels or features
```

Then run the notebooks in order (01, 03, 04, 05); notebook 02 is the steps 1–2 code written out inline.

## Layout

```
src/           download, cleaning (data.py), churn labels (labels.py), features (features.py),
               models, metrics and profit rules (evaluation.py)
tests/         unit tests
notebooks/     01 clean and define churn, 02 steps 1-2 inline, 03 features,
               04 benchmarks/models/evaluation, 05 profit analysis
reports/       report.md (the write-up) and figures/
data/          gitignored; see data/README.md
```
