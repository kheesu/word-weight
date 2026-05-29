"""Compute Residual Information Friction (RIF)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.metrics.expected_cost_model import fit, predict, coefficients


def compute_rif(
    df: pd.DataFrame,
    tfidf_col: str = "tfidf_top_k_mean",
    model_type: str = "linear",
) -> pd.DataFrame:
    """Add expected_cost, residual_cost, positive_residual, and RIF columns."""
    df = df.copy()
    pipe, feat_cols, vif_df = fit(df, model_type)
    df["expected_cost"] = predict(pipe, df, feat_cols)
    df["residual_cost"] = df["token_count"] - df["expected_cost"]
    df["positive_residual"] = df["residual_cost"].clip(lower=0)
    df["RIF"] = df[tfidf_col] * df["positive_residual"]
    return df, pipe, feat_cols


if __name__ == "__main__":
    import argparse, yaml  # noqa: E401

    p = argparse.ArgumentParser()
    p.add_argument("--size", default="small")
    p.add_argument("--tokenizer", default="custom_bpe")
    args = p.parse_args()

    with open(ROOT / "configs/experiments.yaml") as f:
        exp_cfg = yaml.safe_load(f)

    metrics_path = ROOT / "data/processed" / f"word_metrics_{args.size}.parquet"
    df = pd.read_parquet(metrics_path)
    df = df[df["tokenizer"] == args.tokenizer].copy()

    df, pipe, feat_cols = compute_rif(df, model_type=exp_cfg["residual_model"])

    coefs = coefficients(pipe, feat_cols)
    print("\nModel coefficients:")
    print(coefs.to_string(index=False))

    coefs.to_csv(ROOT / "results/residual_model_coefficients.csv", index=False)

    top = df.nlargest(30, "RIF")[["word", "freq", "tfidf_top_k_mean", "token_count",
                                   "expected_cost", "positive_residual", "RIF"]]
    print("\nTop RIF words:")
    print(top.to_string(index=False))
    top.to_csv(ROOT / "results/top_rif_words.csv", index=False)
