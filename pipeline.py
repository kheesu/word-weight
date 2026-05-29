"""Main pipeline runner.

Usage:
  python pipeline.py                  # run all steps, small corpus, custom_bpe
  python pipeline.py --size medium
  python pipeline.py --steps 1 2 3   # run specific steps only
  python pipeline.py --list           # show steps
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

STEPS = {
    1: "preprocess corpus",
    2: "train BPE",
    3: "compute TF-IDF scores",
    4: "compute token costs",
    5: "merge word metrics + IWF",
    6: "exp01 distribution analysis",
    7: "exp02 quadrant analysis",
    8: "exp03 residual friction",
    9: "exp04 vocab simulation",
    10: "exp05 tokenizer comparison",
    11: "plot distributions",
    12: "plot quadrants",
    13: "plot savings curves",
}


def run_pipeline(size: str, steps: list[int], tokenizer: str):
    import yaml  # noqa: PLC0415

    with open(ROOT / "configs/tokenizers.yaml") as f:
        tok_cfg = yaml.safe_load(f)

    bpe_vs = tok_cfg["custom_bpe"]["default_vocab_size"]
    bpe_path = ROOT / tok_cfg["custom_bpe"]["model_dir"] / f"bpe_{size}_{bpe_vs}.json"

    def step(n: int):
        return not steps or n in steps

    if step(1):
        print(f"\n{'='*60}\nStep 1: preprocess corpus ({size})\n{'='*60}")
        from src.preprocessing.build_corpus import build
        build(size)

    if step(2):
        print(f"\n{'='*60}\nStep 2: train BPE\n{'='*60}")
        from src.tokenization.train_bpe import train
        train(size)

    if step(3):
        print(f"\n{'='*60}\nStep 3: compute TF-IDF scores\n{'='*60}")
        from src.tfidf.aggregate_word_scores import build_word_metrics
        build_word_metrics(size)

    if step(4):
        print(f"\n{'='*60}\nStep 4: compute token costs\n{'='*60}")
        import json  # noqa: PLC0415
        from src.tokenization.load_tokenizers import load_custom_bpe, load_gpt2, load_bert
        from src.tokenization.compute_token_costs import compute_for_vocab, load_vocab_words

        with open(ROOT / "configs/wikipedia.yaml") as f:
            wiki_cfg = yaml.safe_load(f)

        vocab_path = ROOT / wiki_cfg["paths"]["processed"] / f"vocab_{size}.json"
        words = load_vocab_words(vocab_path)
        tokenizers = [load_custom_bpe(str(bpe_path)), load_gpt2(), load_bert()]

        # Optionally add SentencePiece trained on the raw corpus
        try:
            from src.tokenization.load_tokenizers import load_sentencepiece
            raw_path = ROOT / wiki_cfg["paths"]["processed"] / f"raw_docs_{size}.jsonl"
            with open(raw_path) as fh:
                sentences = [json.loads(l) for l in fh.readlines()[:5000]]
            tokenizers.append(load_sentencepiece(train_sentences=sentences))
        except Exception as e:
            print(f"[warn] SentencePiece skipped: {e}")

        out = ROOT / "data/processed" / f"token_costs_{size}.parquet"
        compute_for_vocab(words, tokenizers, out)

    if step(5):
        print(f"\n{'='*60}\nStep 5: merge metrics + IWF\n{'='*60}")
        import pandas as pd  # noqa: PLC0415
        from src.tfidf.aggregate_word_scores import build_word_metrics
        from src.tokenization.compute_token_costs import load_vocab_words
        from src.metrics.information_weighted_fragmentation import full_word_table, add_percentile_bins

        with open(ROOT / "configs/wikipedia.yaml") as f:
            wiki_cfg = yaml.safe_load(f)
        with open(ROOT / "configs/experiments.yaml") as f:
            exp_cfg = yaml.safe_load(f)

        proc = ROOT / wiki_cfg["paths"]["processed"]
        tfidf_df = pd.read_parquet(proc / f"tfidf_scores_{size}.parquet")
        cost_df = pd.read_parquet(ROOT / "data/processed" / f"token_costs_{size}.parquet")

        full = full_word_table(tfidf_df, cost_df)
        full = pd.concat(
            [add_percentile_bins(g) for _, g in full.groupby("tokenizer")],
            ignore_index=True,
        )

        out = ROOT / "data/processed" / f"word_metrics_{size}.parquet"
        full.to_parquet(out, index=False)
        print(f"Saved merged word metrics → {out}")

    if step(6):
        print(f"\n{'='*60}\nStep 6: distribution analysis\n{'='*60}")
        from src.experiments.exp01_distribution_analysis import run
        run(size, tokenizer)

    if step(7):
        print(f"\n{'='*60}\nStep 7: quadrant analysis\n{'='*60}")
        from src.experiments.exp02_quadrant_analysis import run
        run(size, tokenizer)

    if step(8):
        print(f"\n{'='*60}\nStep 8: residual friction\n{'='*60}")
        import yaml  # noqa: PLC0415
        with open(ROOT / "configs/experiments.yaml") as f:
            exp_cfg = yaml.safe_load(f)
        from src.experiments.exp03_residual_friction import run
        run(size, tokenizer, exp_cfg["residual_model"])

    if step(9):
        print(f"\n{'='*60}\nStep 9: vocab simulation\n{'='*60}")
        from src.experiments.exp04_vocab_simulation import run
        run(size, tokenizer)

    if step(10):
        print(f"\n{'='*60}\nStep 10: tokenizer comparison\n{'='*60}")
        from src.experiments.exp05_tokenizer_comparison import run
        run(size)

    if step(11):
        print(f"\n{'='*60}\nStep 11: plot distributions\n{'='*60}")
        from src.visualization.plot_distributions import run
        run(size, tokenizer)

    if step(12):
        print(f"\n{'='*60}\nStep 12: plot quadrants\n{'='*60}")
        from src.visualization.plot_quadrants import run
        run(size, tokenizer)

    if step(13):
        print(f"\n{'='*60}\nStep 13: plot savings curves\n{'='*60}")
        from src.visualization.plot_savings_curves import run
        run(tokenizer)

    print("\nPipeline complete.")


def main():
    p = argparse.ArgumentParser(description="Information-Weighted Tokenization Friction pipeline")
    p.add_argument("--size", default="small", choices=["small", "medium", "large"])
    p.add_argument("--tokenizer", default="custom_bpe")
    p.add_argument("--steps", nargs="*", type=int)
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    if args.list:
        for n, desc in STEPS.items():
            print(f"  {n:2d}. {desc}")
        return

    run_pipeline(args.size, args.steps or [], args.tokenizer)


if __name__ == "__main__":
    main()
