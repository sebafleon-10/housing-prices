# What is this house worth? Ames house prices, audited and rebuilt

Sebastian Leon. A rebuild of my BUS 310 (Business Analytics III, DePauw) housing-price project.

## The audit: half the prices were placeholders
The course file has 2,919 rows: the 1,460 sales from Kaggle's House Prices training set (Ames, Iowa, 2006 to 2010) stacked on 1,459 rows from its test set, whose real prices are hidden. Those rows carry placeholder prices instead.

| Rows | Mean price | SD | Correlation of log price with quality |
|---|---|---|---|
| Ids 1 to 1,460 (real sales) | $180,921 | $79,443 | 0.82 |
| Ids 1,461 to 2,919 (placeholders) | $179,184 | $16,518 | 0.10 |

The same six-input regression explains 44 percent of log-price variation on the mixed file and 80 percent on the real sales alone. Everything below uses only the real sales, minus 4 houses over 4,000 sq ft (the dataset author's recommendation), leaving 1,456.

## The model ladder
Repeated 5-fold cross-validation (3 x 5 folds) on log price, with imputation, scaling and encoding fit inside each fold.

| Model | RMSE (log) | R-squared | Mean abs. error | Median % error |
|---|---|---|---|---|
| Class Model 1 (6 inputs) | 0.164 | 0.829 | $20,818 | 9.5% |
| All 79 inputs, plain OLS | 0.122 | 0.905 | $14,183 | 5.9% |
| Ridge | 0.115 | 0.916 | $13,550 | 5.5% |
| **Lasso** | **0.113** | **0.919** | **$13,307** | **5.5%** |
| Random forest | 0.134 | 0.886 | $16,154 | 6.1% |
| LightGBM | 0.119 | 0.910 | $14,250 | 5.7% |
| Blend: Lasso + LightGBM | 0.110 | | $12,908 | 5.2% |

- Out of time: trained on 2006 to 2009 and tested on the 175 sales of 2010, Lasso's median error is 6.5 percent (Model 1: 9.8 percent).
- On real data the 79-input model beats the 6-input model; the original conclusion that it overfit came from the placeholder rows.
- One step of overall quality is worth about 605 sq ft of living area (not 225).
- Model 1 overprices the cheapest fifth of homes by a median 7.8 percent and underprices the top fifth by 5.6 percent; the blend cuts that to +3.0 and -2.5 percent.
- With 1,456 houses and mostly log-linear relationships, regularized regression beats the tree models.

## Run it
Download `train.csv` from [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) or use the course file, saved as `data/raw/housing_price_full_sample.csv` (the scripts use Ids 1 to 1,460 only).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
brew install libomp   # macOS, for LightGBM
PYTHONPATH=src .venv/bin/python src/housing.py
PYTHONPATH=src .venv/bin/python src/time_holdout.py
```

Data: Dean De Cock's Ames Housing data as published in the Kaggle competition. Not redistributed here.
