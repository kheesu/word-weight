"""Compute IWF and merge word-level TF-IDF with token cost data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def merge_word_metrics(
    tfidf_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    tokenizer_name: str,
    tfidf_col: str = "tfidf_top_k_mean",
) -> pd.DataFrame:
    """Join TF-IDF scores with token costs for one tokenizer, compute IWF."""
    costs = cost_df[cost_df["tokenizer"] == tokenizer_name][
        ["word", "token_count", "tokens_per_char", "is_split"]
    ].copy()

    merged = tfidf_df.merge(costs, on="word", how="inner")
    merged["IWF"] = merged[tfidf_col] * (merged["token_count"] - 1).clip(lower=0)
    merged["tokenizer"] = tokenizer_name
    return merged


def full_word_table(
    tfidf_df: pd.DataFrame,
    cost_df: pd.DataFrame,
    tfidf_col: str = "tfidf_top_k_mean",
) -> pd.DataFrame:
    """Build one merged table per tokenizer, stack them."""
    tokenizers = cost_df["tokenizer"].unique()
    frames = [
        merge_word_metrics(tfidf_df, cost_df, t, tfidf_col)
        for t in tokenizers
    ]
    return pd.concat(frames, ignore_index=True)


def add_percentile_bins(df: pd.DataFrame, high_pct: float = 90.0) -> pd.DataFrame:
    """Add quadrant labels based on TF-IDF and token cost percentiles."""
    df = df.copy()
    thr_tfidf = np.percentile(df["tfidf_top_k_mean"], high_pct)
    thr_cost = np.percentile(df["token_count"], high_pct)

    hi_info = df["tfidf_top_k_mean"] >= thr_tfidf
    hi_cost = df["token_count"] >= thr_cost

    conditions = [
        hi_info & hi_cost,
        hi_info & ~hi_cost,
        ~hi_info & hi_cost,
        ~hi_info & ~hi_cost,
    ]
    labels = [
        "high_info_high_cost",
        "high_info_low_cost",
        "low_info_high_cost",
        "low_info_low_cost",
    ]
    df["quadrant"] = np.select(conditions, labels, default="low_info_low_cost")
    df["tfidf_percentile"] = pd.cut(
        df["tfidf_top_k_mean"],
        bins=10,
        labels=[f"d{i}" for i in range(1, 11)],
    )
    return df


if __name__ == "__main__":
    import argparse, yaml  # noqa: E401

    p = argparse.ArgumentParser()
    p.add_argument("--size", default="small")
    args = p.parse_args()

    with open(ROOT / "configs/wikipedia.yaml") as f:
        wiki_cfg = yaml.safe_load(f)

    proc = ROOT / wiki_cfg["paths"]["processed"]
    tfidf_df = pd.read_parquet(proc / f"tfidf_scores_{args.size}.parquet")
    cost_df = pd.read_parquet(ROOT / "data/processed" / f"token_costs_{args.size}.parquet")

    full = full_word_table(tfidf_df, cost_df)
    full = pd.concat(
        [add_percentile_bins(g) for _, g in full.groupby("tokenizer")],
        ignore_index=True,
    )

    out = ROOT / "data/processed" / f"word_metrics_{args.size}.parquet"
    full.to_parquet(out, index=False)
    print(f"Saved merged word metrics → {out}")
    print(full.groupby("tokenizer")["IWF"].describe())
