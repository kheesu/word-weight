"""Experiment 1: Distribution analysis — TF-IDF vs token count correlations."""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results/distributions"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run(size: str = "small", tokenizer: str = "custom_bpe"):
    df = pd.read_parquet(ROOT / "data/processed" / f"word_metrics_{size}.parquet")
    df = df[df["tokenizer"] == tokenizer].copy()

    rows = []
    for col in ["tfidf_top_k_mean", "tfidf_max", "tfidf_mean"]:
        x = df[col].values
        y = df["token_count"].values
        pearson_r, pearson_p = stats.pearsonr(x, y)
        spearman_r, spearman_p = stats.spearmanr(x, y)

        # Partial correlation controlling for word_len
        resid_x = stats.linregress(df["word_len"], x)
        resid_y = stats.linregress(df["word_len"], y)
        x_resid = x - (resid_x.slope * df["word_len"] + resid_x.intercept)
        y_resid = y - (resid_y.slope * df["word_len"] + resid_y.intercept)
        partial_len_r, partial_len_p = stats.spearmanr(x_resid, y_resid)

        # Partial correlation controlling for log_freq
        resid_x2 = stats.linregress(df["log_freq"], x)
        resid_y2 = stats.linregress(df["log_freq"], y)
        x_resid2 = x - (resid_x2.slope * df["log_freq"] + resid_x2.intercept)
        y_resid2 = y - (resid_y2.slope * df["log_freq"] + resid_y2.intercept)
        partial_freq_r, partial_freq_p = stats.spearmanr(x_resid2, y_resid2)

        rows.append({
            "tfidf_metric": col,
            "tokenizer": tokenizer,
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 4),
            "spearman_r": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 4),
            "partial_length_r": round(partial_len_r, 4),
            "partial_length_p": round(partial_len_p, 4),
            "partial_freq_r": round(partial_freq_r, 4),
            "partial_freq_p": round(partial_freq_p, 4),
        })

    corr_df = pd.DataFrame(rows)
    out = ROOT / "results/correlations.csv"
    corr_df.to_csv(out, index=False)
    print(f"Saved correlations → {out}")
    print(corr_df.to_string(index=False))

    # Token count by TF-IDF decile
    df["tfidf_decile"] = pd.qcut(df["tfidf_top_k_mean"], q=10, labels=False, duplicates="drop") + 1
    decile_stats = df.groupby("tfidf_decile")["token_count"].agg(["mean", "median", "std"])
    decile_stats.to_csv(OUT_DIR / f"token_cost_by_tfidf_decile_{tokenizer}.csv")
    print(f"\nToken cost by TF-IDF decile:\n{decile_stats}")

    return corr_df


if __name__ == "__main__":
    import argparse  # noqa

    p = argparse.ArgumentParser()
    p.add_argument("--size", default="small")
    p.add_argument("--tokenizer", default="custom_bpe")
    args = p.parse_args()
    run(args.size, args.tokenizer)
