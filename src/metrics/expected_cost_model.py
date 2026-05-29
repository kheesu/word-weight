"""Fit a Poisson regression model predicting expected token count from word features."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# log_mean_tf = log(freq / doc_freq) replaces log_doc_freq.
# Motivation: log_freq and log_doc_freq have r ≈ 0.95 (VIF ≈ 11), which
# destabilises coefficient estimates. log_mean_tf = log_freq − log_doc_freq
# captures word burstiness (avg occurrences per document) and is nearly
# orthogonal to log_freq (r ≈ 0.2 on this corpus).
FEATURES = ["word_len", "log_freq", "log_mean_tf"]


def _add_char_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_mean_tf"] = (df["log_freq"] - df["log_doc_freq"]).clip(lower=0)
    df["n_digits"] = df["word"].str.count(r"\d")
    df["n_upper"] = df["word"].str.count(r"[A-Z]")
    df["has_hyphen"] = df["word"].str.contains("-").astype(int)
    df["has_nonascii"] = df["word"].apply(lambda w: int(any(ord(c) > 127 for c in w)))
    return df


def _check_vif(X_scaled: np.ndarray, feat_cols: list[str], threshold: float = 10.0):
    rows = []
    for i, col in enumerate(feat_cols):
        try:
            vif = variance_inflation_factor(X_scaled, i)
        except Exception:
            vif = float("nan")
        rows.append({"feature": col, "VIF": round(vif, 2)})
        if vif > threshold:
            print(f"[warn] High VIF for '{col}': {vif:.1f} (threshold {threshold})")
    return pd.DataFrame(rows)


def fit(df: pd.DataFrame, model_type: str = "poisson") -> tuple:
    """Return (fitted_pipeline, feature_names, vif_df)."""
    df = _add_char_features(df)
    feat_cols = FEATURES + ["n_digits", "n_upper", "has_hyphen", "has_nonascii"]
    feat_cols = [c for c in feat_cols if c in df.columns]

    X = df[feat_cols].fillna(0).values
    y = df["token_count"].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    vif_df = _check_vif(X_scaled, feat_cols)

    reg = PoissonRegressor(max_iter=500)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", reg)])
    pipe.fit(X, y)
    return pipe, feat_cols, vif_df


def predict(pipe, df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    df = _add_char_features(df)
    X = df[feat_cols].fillna(0).values
    return pipe.predict(X)


def coefficients(pipe, feat_cols: list[str]) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    if not hasattr(model, "coef_"):
        return pd.DataFrame()
    coef = model.coef_
    return pd.DataFrame({
        "feature": feat_cols,
        "coef": coef,
        "exp_coef": np.exp(coef),  # multiplicative effect per SD of the feature
    })
