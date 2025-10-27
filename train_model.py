# train_model.py
# Training with log-target + HistGradientBoostingRegressor, no 'locality' feature.

import time
print("Starting training script...", flush=True)

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib

DATA_PATH = Path("Mumbai House Prices.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# 1) Load
df = pd.read_csv(DATA_PATH)
print("CSV loaded", flush=True)

# 2) Expected columns
expected = ["bhk", "type", "area", "price", "price_unit", "region", "status", "age"]
missing = [c for c in expected if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# 3) Cleaning
df = df.dropna(subset=["area", "price", "bhk", "region"])
df = df[(df["area"] > 0) & (df["price"] > 0) & (df["bhk"] > 0)]

def to_inr(row):
    val = row["price"]
    unit = str(row["price_unit"]).strip().lower()
    if unit.startswith("cr"):
        return val * 1e7
    if unit.startswith("l"):
        return val * 1e5
    return val

df["price_inr"] = df.apply(to_inr, axis=1)

def iqr_trim(s, k=3.0):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return lo, hi

for col in ["price_inr", "area"]:
    lo, hi = iqr_trim(df[col], k=3.0)
    df = df[(df[col] >= lo) & (df[col] <= hi)]

print(f"Rows after cleaning: {len(df):,}", flush=True)

# 4) Features/target (no 'locality')
y = df["price_inr"]
X = df[["area", "bhk", "type", "region", "status", "age"]].copy()

# Age handling
if X["age"].dtype == object:
    X["age_cat"] = X["age"].astype(str)
    X["age_num"] = 0.0
    cat_cols = ["type", "region", "status", "age_cat"]
    num_cols = ["area", "bhk", "age_num"]
else:
    cat_cols = ["type", "region", "status"]
    num_cols = ["area", "bhk", "age"]

print("Unique counts:",
      "type", X["type"].nunique(),
      "region", X["region"].nunique(),
      "status", X["status"].nunique(),
      flush=True)

# 5) Preprocessing
pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

# 6) Use log-target to stabilize and improve fit
def log1p_transform(y_arr):
    return np.log1p(y_arr)

def expm1_inverse(y_arr):
    return np.expm1(y_arr)

log_y = FunctionTransformer(func=log1p_transform, inverse_func=expm1_inverse, validate=False)

# 7) Model: HistGradientBoostingRegressor (fast and accurate on tabular)
hgb = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=10,
    max_bins=255,
    l2_regularization=1.0,
    early_stopping=True,
    random_state=42
)

# Full pipeline: pre -> log-target via TransformedTargetRegressor-like pattern
# Since older sklearn may lack TransformedTargetRegressor with HGBR conveniences,
# we manually transform y for training and invert for metrics and saving.
pipe = Pipeline(steps=[("pre", pre), ("model", hgb)])

# 8) Split, fit on log-target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train/Test shapes: {X_tr.shape} / {X_te.shape}", flush=True)

print("Fitting model...", flush=True)
t0 = time.time()
pipe.fit(X_tr, np.log1p(y_tr))
print(f"Fit time: {time.time()-t0:.1f}s", flush=True)

# 9) Predict and invert log
pred_log = pipe.predict(X_te)
pred = np.expm1(pred_log)

# 10) Metrics
r2 = r2_score(y_te, pred)
mae = mean_absolute_error(y_te, pred)
mse = mean_squared_error(y_te, pred)
mse = mean_squared_error(y_te, pred)
rmse = float(np.sqrt(mse))



print("Evaluation on test set:")
print(f"R²: {r2:.4f}")
print(f"MAE (₹): {mae:,.2f}")
print(f"RMSE (₹): {rmse:,.2f}")


# 11) Save pipeline + metadata
joblib.dump(pipe, MODEL_DIR / "price_model.pkl")

def uniq(series):
    return sorted(pd.Series(series).astype(str).dropna().unique().tolist())

meta = {
    "regions": uniq(X["region"]),
    "types": uniq(X["type"]),
    "statuses": uniq(X["status"]),
    "bhk_min": int(df["bhk"].min()),
    "bhk_max": int(df["bhk"].max()),
    "area_min": int(df["area"].min()),
    "area_max": int(df["area"].max()),
    "uses_log_target": True  # app will invert predictions
}
joblib.dump(meta, MODEL_DIR / "metadata.pkl")
print("Artifacts saved to models/.", flush=True)
