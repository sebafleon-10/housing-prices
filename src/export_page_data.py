"""Fit the small in-browser estimator and export everything the portfolio page needs.
Usage: PYTHONPATH=src python src/export_page_data.py OUT.json (run housing.py and time_holdout.py first)."""
import json, sys
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import RepeatedKFold
from prep import load

raw, df, _ = load()
res = json.load(open("results/housing.json"))
y = np.log(df.SalePrice.to_numpy())

# Estimator: five numeric inputs plus neighborhood, plain OLS on log price.
NUMS = ["GrLivArea", "OverallQual", "YearBuilt", "TotalBsmtSF", "GarageCars"]
med = df[NUMS].median()
base_nb = df.Neighborhood.value_counts().idxmax()   # most common neighborhood is the reference level
def design(d):
    X = d[NUMS].fillna(med).astype(float)
    nb = pd.get_dummies(d.Neighborhood, dtype=float)
    nb = nb.reindex(columns=sorted(df.Neighborhood.unique()), fill_value=0.0).drop(columns=[base_nb])
    return sm.add_constant(pd.concat([X, nb], axis=1), has_constant="add")
X = design(df)
pct_err = []
for a, b in RepeatedKFold(n_splits=5, n_repeats=3, random_state=42).split(df):
    m = sm.OLS(y[a], X.iloc[a]).fit()
    p = np.exp(m.predict(X.iloc[b])); pct_err.extend(np.abs(p - df.SalePrice.iloc[b]) / df.SalePrice.iloc[b])
pct_err = np.array(pct_err)
fit = sm.OLS(y, X).fit()
nb_effect = {base_nb: 0.0, **{k: float(fit.params[k]) for k in X.columns if k not in ["const"] + NUMS}}
nb_median = df.groupby("Neighborhood").SalePrice.median()
neighborhoods = sorted(({"code": k, "effect": v, "median": float(nb_median[k]), "n": int((df.Neighborhood == k).sum())}
                        for k, v in nb_effect.items()), key=lambda r: r["median"])
estimator = {"const": float(fit.params["const"]), "coef": {k: float(fit.params[k]) for k in NUMS},
             "neighborhoods": neighborhoods, "base": base_nb,
             "ranges": {k: [float(df[k].quantile(0.01)), float(df[k].median()), float(df[k].quantile(0.99))] for k in NUMS},
             "medianPctError": float(np.median(pct_err)), "p80PctError": float(np.quantile(pct_err, 0.8)), "r2": float(fit.rsquared)}
print("estimator median err", round(estimator["medianPctError"], 4), "80% within", round(estimator["p80PctError"], 4), "R2", round(fit.rsquared, 3))

# Histogram of the class file, split into real and placeholder rows (shared $20K bins to $500K).
bins = np.arange(0, 520_000, 20_000)
real, fake = raw[raw.Id <= 1460].SalePrice, raw[raw.Id > 1460].SalePrice
hist = {"edges": bins.tolist(), "real": np.histogram(real.clip(upper=499_999), bins)[0].tolist(),
        "placeholder": np.histogram(fake.clip(upper=499_999), bins)[0].tolist()}

nice = {"GrLivArea": "Living area", "OverallQual": "Overall quality", "YearBuilt": "Year built", "TotalBsmtSF": "Basement size",
        "OverallCond": "Overall condition", "1stFlrSF": "First-floor area", "BsmtFinSF1": "Finished basement", "YearRemodAdd": "Year remodeled",
        "ExterQual": "Exterior quality", "FullBath": "Full bathrooms", "GarageCars": "Garage size", "LotArea": "Lot size"}
out = {"audit": res["audit"], "n": res["n"], "droppedBig": res["dropped_big"], "hist": hist,
       "ladder": [{"model": r["model"], "medianPctError": r["median_pct_error"], "r2": r["r2"], "mae": r["mae"]} for r in res["ladder"]]
                 + [{"model": "Blend: Lasso + LightGBM", "medianPctError": res["blend"]["median_pct_error"], "r2": None, "mae": res["blend"]["mae"]}],
       "holdout": res["time_holdout"], "bias": res["bias_by_band"], "model1Real": res["model1_real"],
       "shap": [{"feature": nice.get(k, k), "value": v} for k, v in res["shap"].items()][:8], "estimator": estimator}
json.dump(out, open(sys.argv[1], "w"), separators=(",", ":"), default=float)
print("neighborhoods", len(neighborhoods), "base", base_nb)
