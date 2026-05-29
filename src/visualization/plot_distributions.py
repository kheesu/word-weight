"""Figures 1 and 3: TF-IDF vs token count scatter and decile bar chart."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_scatter(df: pd.DataFrame, tokenizer: str, sample: int = 5000):
    sub = df.sample(min(sample, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        sub["tfidf_top_k_mean"],
        sub["token_count"],
        c=sub["IWF"],
        cmap="viridis",
        alpha=0.4,
        s=8,
        rasterized=True,
    )
    plt.colorbar(sc, ax=ax, label="IWF")
    ax.set_xlabel("TF-IDF importance (top-k mean)")
    ax.set_ylabel("Token count")
    ax.set_title(f"TF-IDF vs Token Count — {tokenizer}")
    fig.tight_layout()
    out = FIG_DIR / f"tfidf_vs_token_count_{tokenizer}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


def plot_decile_bar(size: str, tokenizer: str):
    csv_path = ROOT / "results/distributions" / f"token_cost_by_tfidf_decile_{tokenizer}.csv"
    if not csv_path.exists():
        print(f"Missing {csv_path}, run exp01 first")
        return
    dec = pd.read_csv(csv_path, index_col=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(dec.index, dec["mean"], yerr=dec["std"], capsize=3, color="steelblue")
    ax.set_xlabel("TF-IDF decile")
    ax.set_ylabel("Mean token count")
    ax.set_title(f"Token count by TF-IDF decile — {tokenizer}")
    fig.tight_layout()
    out = FIG_DIR / f"token_count_by_tfidf_decile_{tokenizer}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


def run(size: str = "small", tokenizer: str = "custom_bpe"):
    df = pd.read_parquet(ROOT / "data/processed" / f"word_metrics_{size}.parquet")
    df = df[df["tokenizer"] == tokenizer]
    plot_scatter(df, tokenizer)
    plot_decile_bar(size, tokenizer)


if __name__ == "__main__":
    import argparse  # noqa

    p = argparse.ArgumentParser()
    p.add_argument("--size", default="small")
    p.add_argument("--tokenizer", default="custom_bpe")
    args = p.parse_args()
    run(args.size, args.tokenizer)
