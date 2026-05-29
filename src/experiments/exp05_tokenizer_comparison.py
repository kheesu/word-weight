"""Experiment 5: Cross-tokenizer friction comparison."""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.metrics.residual_friction import compute_rif
from src.experiments.exp04_vocab_simulation import simulate


def run(size: str = "small"):
    import yaml  # noqa: PLC0415

    with open(ROOT / "configs/experiments.yaml") as f:
        exp_cfg = yaml.safe_load(f)

    df_all = pd.read_parquet(ROOT / "data/processed" / f"word_metrics_{size}.parquet")
    tokenizers = df_all["tokenizer"].unique()

    summary_rows = []
    for tok in tokenizers:
        df = df_all[df_all["tokenizer"] == tok].copy()
        df, _, _ = compute_rif(df, model_type=exp_cfg["residual_model"])

        thr_tfidf = df["tfidf_top_k_mean"].quantile(0.9)
        hi_info = df[df["tfidf_top_k_mean"] >= thr_tfidf]

        sim = simulate(df, [1000])
        rif_savings = sim[sim["strategy"] == "RIF"]["information_weighted_savings"].iloc[0]

        # Top-1k RIF words
        top_rif_words = set(df.nlargest(1000, "RIF")["word"])

        summary_rows.append({
            "tokenizer": tok,
            "mean_word_cost": round(df["token_count"].mean(), 3),
            "median_word_cost": round(df["token_count"].median(), 3),
            "split_rate": round(df["is_split"].mean(), 4),
            "high_info_split_rate": round(hi_info["is_split"].mean(), 4),
            "mean_RIF": round(df["RIF"].mean(), 4),
            "top_1k_info_savings": round(rif_savings, 4),
        })

    summary = pd.DataFrame(summary_rows)
    out = ROOT / "results/tokenizer_comparison.csv"
    summary.to_csv(out, index=False)
    print(f"Saved tokenizer comparison → {out}")
    print(summary.to_string(index=False))

    # Top-k RIF overlap across tokenizers
    rif_sets = {}
    for tok in tokenizers:
        df = df_all[df_all["tokenizer"] == tok].copy()
        df, _, _ = compute_rif(df, model_type=exp_cfg["residual_model"])
        rif_sets[tok] = set(df.nlargest(1000, "RIF")["word"])

    tok_list = list(rif_sets.keys())
    overlap_rows = []
    for i, t1 in enumerate(tok_list):
        for t2 in tok_list[i + 1:]:
            overlap = len(rif_sets[t1] & rif_sets[t2]) / 1000
            overlap_rows.append({"tok_a": t1, "tok_b": t2, "top1k_rif_overlap": round(overlap, 3)})

    overlap_df = pd.DataFrame(overlap_rows)
    overlap_df.to_csv(ROOT / "results/tokenizer_rif_overlap.csv", index=False)
    print(f"\nTop-1k RIF overlap:\n{overlap_df.to_string(index=False)}")
    return summary


if __name__ == "__main__":
    import argparse  # noqa

    p = argparse.ArgumentParser()
    p.add_argument("--size", default="small")
    args = p.parse_args()
    run(args.size)
