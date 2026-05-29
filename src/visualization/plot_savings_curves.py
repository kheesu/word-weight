"""Figure 5: Vocabulary simulation savings curves."""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_STYLES = {
    "frequency": ("C0", "-"),
    "tfidf": ("C1", "--"),
    "token_cost": ("C2", "-."),
    "IWF": ("C3", ":"),
    "RIF": ("C4", "-"),
    "random": ("gray", "--"),
}


def _plot(df: pd.DataFrame, metric: str, title: str, out: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for strategy, (color, ls) in STRATEGY_STYLES.items():
        sub = df[df["strategy"] == strategy].sort_values("k")
        if sub.empty:
            continue
        ax.plot(sub["k"], sub[metric], label=strategy, color=color, ls=ls, marker="o", ms=4)
    ax.set_xlabel("Vocabulary additions (K)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


def run(tokenizer: str = "custom_bpe"):
    csv = ROOT / "results/vocab_simulation" / f"savings_{tokenizer}.csv"
    if not csv.exists():
        print(f"Missing {csv}, run exp04 first")
        return
    df = pd.read_csv(csv)

    _plot(df, "corpus_token_savings", f"Corpus Token Savings — {tokenizer}",
          FIG_DIR / f"token_savings_curve_{tokenizer}.png")
    _plot(df, "information_weighted_savings", f"Information-Weighted Savings — {tokenizer}",
          FIG_DIR / f"information_savings_curve_{tokenizer}.png")


if __name__ == "__main__":
    import argparse  # noqa

    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="custom_bpe")
    args = p.parse_args()
    run(args.tokenizer)
