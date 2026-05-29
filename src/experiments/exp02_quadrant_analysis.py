"""Experiment 2: Quadrant analysis — classify words by info value and token cost."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

QUAD_DIR = ROOT / "results/quadrants"
QUAD_DIR.mkdir(parents=True, exist_ok=True)


def run(size: str = "small", tokenizer: str = "custom_bpe", high_pct: float = 90.0):
    df = pd.read_parquet(ROOT / "data/processed" / f"word_metrics_{size}.parquet")
    df = df[df["tokenizer"] == tokenizer].copy()

    thr_tfidf = df["tfidf_top_k_mean"].quantile(high_pct / 100)
    thr_cost = df["token_count"].quantile(high_pct / 100)

    quadrants = {
        "high_info_high_cost": df[
            (df["tfidf_top_k_mean"] >= thr_tfidf) & (df["token_count"] >= thr_cost)
        ],
        "high_info_low_cost": df[
            (df["tfidf_top_k_mean"] >= thr_tfidf) & (df["token_count"] < thr_cost)
        ],
        "low_info_high_cost": df[
            (df["tfidf_top_k_mean"] < thr_tfidf) & (df["token_count"] >= thr_cost)
        ],
        "low_info_low_cost": df[
            (df["tfidf_top_k_mean"] < thr_tfidf) & (df["token_count"] < thr_cost)
        ],
    }

    cols = ["word", "freq", "doc_freq", "tfidf_top_k_mean", "token_count", "IWF"]
    summary_rows = []
    for name, sub in quadrants.items():
        sub_sorted = sub.sort_values("IWF", ascending=False)
        sub_sorted[cols].to_csv(QUAD_DIR / f"{name}_{tokenizer}.csv", index=False)
        summary_rows.append({
            "quadrant": name,
            "count": len(sub),
            "pct": round(100 * len(sub) / len(df), 1),
            "mean_tfidf": round(sub["tfidf_top_k_mean"].mean(), 4),
            "mean_token_count": round(sub["token_count"].mean(), 3),
            "mean_IWF": round(sub["IWF"].mean(), 4),
        })
        print(f"\n{name} (n={len(sub)}): top 10 words")
        print(sub_sorted["word"].head(10).tolist())

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(QUAD_DIR / f"quadrant_summary_{tokenizer}.csv", index=False)
    print(f"\nQuadrant summary:\n{summary.to_string(index=False)}")
    return summary


if __name__ == "__main__":
    import argparse  # noqa

    p = argparse.ArgumentParser()
    p.add_argument("--size", default="small")
    p.add_argument("--tokenizer", default="custom_bpe")
    p.add_argument("--pct", type=float, default=90.0)
    args = p.parse_args()
    run(args.size, args.tokenizer, args.pct)
