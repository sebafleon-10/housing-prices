"""Shared data prep and model definitions for the Ames rebuild."""
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

SEED = 42
M1 = ["GrLivArea", "OverallQual", "GarageArea", "TotalBsmtSF", "FullBath", "Fireplaces"]
RAW = "data/raw/housing_price_full_sample.csv"


def load():
    """The 1,460 real sales (Ids 1 to 1460) minus houses over 4,000 sq ft, as the dataset's author recommends."""
    raw = pd.read_csv(RAW)
    real = raw[raw.Id <= 1460].copy()
    big = real.GrLivArea > 4000
    df = real[~big].reset_index(drop=True)
    for c in ["MSSubClass", "MoSold", "YrSold"]:
        df[c] = df[c].astype(str)
    return raw, df, int(big.sum())


def build(df):
    X_ALL = [c for c in df.columns if c not in ("Id", "SalePrice")]
    CAT = [c for c in X_ALL if not pd.api.types.is_numeric_dtype(df[c])]
    NUM = [c for c in X_ALL if c not in CAT]

    def lin_prep(num, cat):
        parts = [("num", make_pipeline(SimpleImputer(strategy="mean"), StandardScaler()), num)]
        if cat:
            parts.append(("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore", min_frequency=10)), cat))
        return ColumnTransformer(parts)

    tree_prep = ColumnTransformer([("num", "passthrough", NUM),
                                   ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1), CAT)])
    models = {
        "Class Model 1 (6 inputs)": (make_pipeline(lin_prep(M1, []), LinearRegression()), M1),
        "All 79 inputs, plain OLS": (make_pipeline(lin_prep(NUM, CAT), LinearRegression()), X_ALL),
        "Ridge": (make_pipeline(lin_prep(NUM, CAT), RidgeCV(alphas=np.logspace(-4, 3, 60))), X_ALL),
        "Lasso": (make_pipeline(lin_prep(NUM, CAT), LassoCV(alphas=np.logspace(-4, -1, 40), max_iter=20000, random_state=SEED)), X_ALL),
        "Random forest": (make_pipeline(tree_prep, RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.4, n_jobs=-1, random_state=SEED)), X_ALL),
    }
    return X_ALL, NUM, CAT, tree_prep, models
