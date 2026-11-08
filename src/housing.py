"""Ames house prices, rebuilt.

1. Audit: the class file stacks Kaggle's 1,460 real sales on top of 1,459
   rows whose SalePrice is a placeholder. Only the real sales are used.
2. Model ladder scored with repeated 5-fold cross-validation on log price,
   every preprocessing step fit inside the folds.
"""
import json
import numpy as np, pandas as pd
import lightgbm as lgb
import statsmodels.api as sm
from sklearn.model_selection import RepeatedKFold
from prep import M1, SEED, build, load

raw, df, n_big = load()
real, fake = raw[raw.Id <= 1460].copy(), raw[raw.Id > 1460].copy()

# ---------- 1. Audit ----------
corr = lambda d: float(np.corrcoef(np.log(d.SalePrice), d.OverallQual)[0, 1])
def m1_r2(d):
    X = sm.add_constant(d[M1].fillna(d[M1].mean()))
    return float(sm.OLS(np.log(d.SalePrice), X).fit().rsquared)
audit = {"rows_total": len(raw), "rows_real": len(real), "rows_placeholder": len(fake),
         "real": {"mean": real.SalePrice.mean(), "sd": real.SalePrice.std(), "corr_quality": corr(real)},
         "placeholder": {"mean": fake.SalePrice.mean(), "sd": fake.SalePrice.std(), "corr_quality": corr(fake)},
         "model1_r2_all_rows": m1_r2(raw), "model1_r2_real_rows": m1_r2(real)}
print(json.dumps(audit, indent=1, default=float))

# ---------- 2. Data and models ----------
y = np.log(df.SalePrice.to_numpy())
X_ALL, NUM, CAT, tree_prep, models = build(df)
print(f"{len(df)} real sales after dropping {n_big} houses over 4,000 sq ft; {len(NUM)} numeric, {len(CAT)} categorical")
LGB = dict(objective="regression", learning_rate=0.02, num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.5, reg_lambda=1.0, verbose=-1, seed=SEED)

rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)
oof = {k: np.zeros((3, len(df))) for k in list(models) + ["LightGBM"]}
for f, (a, b) in enumerate(rkf.split(df)):
    rep = f // 5
    for k, (m, cols) in models.items():
        m.fit(df.iloc[a][cols], y[a]); oof[k][rep, b] = m.predict(df.iloc[b][cols])
    tp = tree_prep.fit(df.iloc[a][X_ALL]); Xa, Xb = tp.transform(df.iloc[a][X_ALL]), tp.transform(df.iloc[b][X_ALL])
    cut = int(len(a) * 0.8); rng = np.random.default_rng(SEED + f); perm = rng.permutation(len(a))
    es = lgb.train(LGB, lgb.Dataset(Xa[perm[:cut]], y[a][perm[:cut]]), 5000, valid_sets=[lgb.Dataset(Xa[perm[cut:]], y[a][perm[cut:]])],
                   callbacks=[lgb.early_stopping(200, verbose=False)])
    g = lgb.train(LGB, lgb.Dataset(Xa, y[a]), es.best_iteration)
    oof["LightGBM"][rep, b] = g.predict(Xb)
    print(f"fold {f + 1}/15 done", flush=True)

price = df.SalePrice.to_numpy()
ladder = []
for k, P in oof.items():
    rmse = [float(np.sqrt(np.mean((P[r] - y) ** 2))) for r in range(3)]
    r2 = [1 - np.sum((P[r] - y) ** 2) / np.sum((y - y.mean()) ** 2) for r in range(3)]
    mae = [float(np.mean(np.abs(np.exp(P[r]) - price))) for r in range(3)]
    mape = [float(np.median(np.abs(np.exp(P[r]) - price) / price)) for r in range(3)]
    ladder.append({"model": k, "rmse_log": np.mean(rmse), "rmse_sd": np.std(rmse), "r2": float(np.mean(r2)),
                   "mae": np.mean(mae), "median_pct_error": np.mean(mape)})
    print(f"{k:28s} RMSE(log) {np.mean(rmse):.4f} ±{np.std(rmse):.4f}  R² {np.mean(r2):.3f}  MAE ${np.mean(mae):,.0f}  median error {np.mean(mape):.1%}")

# Blend of the two best families (regularized linear + boosting)
blend = (oof["Lasso"] + oof["LightGBM"]) / 2
b_rmse = float(np.mean([np.sqrt(np.mean((blend[r] - y) ** 2)) for r in range(3)]))
b_mae = float(np.mean([np.mean(np.abs(np.exp(blend[r]) - price)) for r in range(3)]))
b_mape = float(np.mean([np.median(np.abs(np.exp(blend[r]) - price) / price) for r in range(3)]))
print(f"Blend lasso + LightGBM         RMSE(log) {b_rmse:.4f}  MAE ${b_mae:,.0f}  median error {b_mape:.1%}")

# Errors by price band: does the bias at the extremes from the class paper survive?
band = pd.qcut(price, 5, labels=["Lowest 20%", "Next 20%", "Middle 20%", "Next 20%+", "Top 20%"])
bias = {}
for k in ["Class Model 1 (6 inputs)", "Lasso"]:
    pct = (np.exp(oof[k].mean(0)) - price) / price
    bias[k] = pd.Series(pct).groupby(band, observed=True).median().round(4).tolist()
bias_b = pd.Series((np.exp(blend.mean(0)) - price) / price).groupby(band, observed=True).median().round(4).tolist()
bias["Blend"] = bias_b
print("median % over(+)/under(-) prediction by price band:", bias)

# Size vs quality on real data (the paper's 225 sq ft trade-off)
fit = sm.OLS(y, sm.add_constant(df[M1].fillna(df[M1].mean()))).fit()
sqft_per_quality = float(fit.params["OverallQual"] / fit.params["GrLivArea"])
print(f"Model 1 on real sales: quality step {fit.params['OverallQual']:.4f}, per sq ft {fit.params['GrLivArea']:.6f} -> one quality step = {sqft_per_quality:.0f} sq ft")

# Drivers: LightGBM SHAP on all data
tp = tree_prep.fit(df[X_ALL]); XA = tp.transform(df[X_ALL])

g_all = lgb.train(LGB, lgb.Dataset(XA, y), 1500)
contrib = g_all.predict(XA, pred_contrib=True)[:, :-1]
shap = pd.Series(np.abs(contrib).mean(0), index=NUM + CAT).sort_values(ascending=False).head(10)
print("top SHAP:", shap.round(4).to_dict())

json.dump({"audit": audit, "n": len(df), "dropped_big": n_big, "ladder": ladder,
           "blend": {"rmse_log": b_rmse, "mae": b_mae, "median_pct_error": b_mape}, "bias_by_band": bias,
           "model1_real": {"quality": fit.params["OverallQual"], "sqft": fit.params["GrLivArea"], "sqft_per_quality": sqft_per_quality, "r2": fit.rsquared},
           "shap": shap.to_dict()}, open("results/housing.json", "w"), indent=2, default=float)
