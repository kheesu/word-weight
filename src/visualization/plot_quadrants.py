"""Figure 2: Quadrant scatter plot."""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "high_info_high_cost": "#d62728",
    "high_info_low_cost": "#2ca02c",
    "low_info_high_cost": "#ff7f0e",
    "low_info_low_cost": "#aec7e8",
}
LABELS = {
    "high_info_high_cost": "information friction",
    "high_info_low_cost": "efficient signal words",
    "low_info_high_cost": "dead-weight fragmentation",
    "low_info_low_cost": "common vocabulary",
}


def run(size: str = "small", tokenizer: str = "custom_bpe", sample: int = 8000):
    df = pd.read_parquet(ROOT / "data/processed" / f"word_metrics_{size}.parquet")
    df = df[df["tokenizer"] == tokenizer].copy()

    # Assign quadrants
    thr_tfidf = df["tfidf_top_k_mean"].quantile(0.90)
    thr_cost = df["token_count"].quantile(0.90)
    hi_info = df["tfidf_top_k_mean"] >= thr_tfidf
    hi_cost = df["token_count"] >= thr_cost
    df["quadrant"] = np.select(
        [hi_info & hi_cost, hi_info & ~hi_cost, ~hi_info & hi_cost],
        ["high_info_high_cost", "high_info_low_cost", "low_info_high_cost"],
        default="low_info_low_cost",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    for q, color in COLORS.items():
        sub = df[df["quadrant"] == q].sample(min(sample // 4, len(df[df["quadrant"] == q])),
                                              random_state=42)
        ax.scatter(
            sub["tfidf_top_k_mean"],
            sub["token_count"],
            c=color,
            alpha=0.4,
            s=6,
            label=LABELS[q],
            rasterized=True,
        )

    ax.axvline(thr_tfidf, color="gray", lw=0.8, ls="--")
    ax.axhline(thr_cost, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("TF-IDF importance")
    ax.set_ylabel("Token count")
    ax.set_title(f"Quadrant Analysis — {tokenizer}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()

    out = FIG_DIR / f"quadrant_scatter_{tokenizer}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    import argparse  # noqa

    p = argparse.ArgumentParser()
    p.add_argument("--size", default="small")
    p.add_argument("--tokenizer", default="custom_bpe")
    args = p.parse_args()
    run(args.size, args.tokenizer)
