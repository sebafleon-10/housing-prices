"""Robustness: train on 2006-2009 sales, test on 2010 (the year after the crash). Run after housing.py."""
import json
import numpy as np
from prep import build, load

_, df, _ = load()
y = np.log(df.SalePrice.to_numpy())
_, _, _, _, models = build(df)
yr = df.YrSold.astype(int).to_numpy(); tr, te = yr <= 2009, yr == 2010
out = {}
for k in ["Class Model 1 (6 inputs)", "Lasso"]:
    m, cols = models[k]
    m.fit(df[tr][cols], y[tr]); p = m.predict(df[te][cols])
    price = df.SalePrice[te].to_numpy()
    out[k] = {"rmse_log": float(np.sqrt(np.mean((p - y[te]) ** 2))), "median_pct_error": float(np.median(np.abs(np.exp(p) - price) / price))}
    print(f"2010 holdout ({te.sum()} sales) {k:26s} RMSE(log) {out[k]['rmse_log']:.4f}  median error {out[k]['median_pct_error']:.1%}")
res = json.load(open("results/housing.json")); res["time_holdout"] = {"n_test": int(te.sum()), "models": out}
json.dump(res, open("results/housing.json", "w"), indent=2, default=float)
