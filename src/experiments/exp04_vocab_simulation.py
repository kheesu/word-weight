"""Experiment 4: Vocabulary adaptation simulation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.metrics.residual_friction import compute_rif

SIM_DIR = ROOT / "results/vocab_simulation"
SIM_DIR.mkdir(parents=True, exist_ok=True)


def simulate(df: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    """Compare vocabulary selection strategies across K values."""
    total_tokens = (df["freq"] * df["token_count"]).sum()
    total_info = (df["tfidf_top_k_mean"] * df["freq"] * df["token_count"]).sum()

    strategies = {
        "random": df.sample(frac=1, random_state=42).reset_index(drop=True),
        "frequency": df.sort_values("freq", ascending=False).reset_index(drop=True),
        "tfidf": df.sort_values("tfidf_top_k_mean", ascending=False).reset_index(drop=True),
        "token_cost": df.sort_values("token_count", ascending=False).reset_index(drop=True),
        "IWF": df.sort_values("IWF", ascending=False).reset_index(drop=True),
        "RIF": df.sort_values("RIF", ascending=False).reset_index(drop=True),
    }

    rows = []
    for k in k_values:
        for name, ranked in strategies.items():
            selected = ranked.head(k)
            # savings_per_occurrence = token_count - 1 (new cost = 1 token)
            savings = selected["freq"] * (selected["token_count"] - 1).clip(lower=0)
            corpus_savings = savings.sum() / total_tokens if total_tokens > 0 else 0.0

            info_savings_num = (
                selected["tfidf_top_k_mean"] * selected["freq"] * (selected["token_count"] - 1).clip(lower=0)
            ).sum()
            info_savings = info_savings_num / total_info if total_info > 0 else 0.0

            rows.append({
                "strategy": name,
                "k": k,
                "corpus_token_savings": round(corpus_savings, 6),
                "information_weighted_savings": round(info_savings, 6),
                "mean_token_count": round(selected["token_count"].mean(), 3),
                "mean_tfidf": round(selected["tfidf_top_k_mean"].mean(), 4),
            })

    return pd.DataFrame(rows)


def run(size: str = "small", tokenizer: str = "custom_bpe"):
    import yaml  # noqa: PLC0415

    with open(ROOT / "configs/experiments.yaml") as f:
        exp_cfg = yaml.safe_load(f)

    df = pd.read_parquet(ROOT / "data/processed" / f"word_metrics_{size}.parquet")
    df = df[df["tokenizer"] == tokenizer].copy()
    df, _, _ = compute_rif(df, model_type=exp_cfg["residual_model"])

    results = simulate(df, exp_cfg["vocab_simulation"]["k_values"])

    results.to_csv(SIM_DIR / f"savings_{tokenizer}.csv", index=False)
    print(results.to_string(index=False))

    pivot = results.pivot(index="k", columns="strategy", values="information_weighted_savings")
    print("\nInformation-weighted savings by K and strategy:")
    print(pivot.to_string())
    return results


if __name__ == "__main__":
    import argparse  # noqa

    p = argparse.ArgumentParser()
    p.add_argument("--size", default="small")
    p.add_argument("--tokenizer", default="custom_bpe")
    args = p.parse_args()
    run(args.size, args.tokenizer)
