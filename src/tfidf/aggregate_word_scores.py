"""Aggregate per-document TF-IDF scores into a single importance score per word."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def aggregate(
    word_scores: dict[str, list[float]],
    vocab: dict[str, dict],
    top_k: int = 5,
) -> pd.DataFrame:
    """
    word_scores: {word: [tfidf_in_each_doc_where_it_appears]}
    vocab:       {word: {freq, doc_freq}}
    Returns a DataFrame with one row per word.
    """
    rows = []
    for word, scores in word_scores.items():
        if word not in vocab:
            continue
        arr = np.array(scores)
        k = min(top_k, len(arr))
        top_k_mean = float(np.mean(np.partition(arr, -k)[-k:]))
        rows.append({
            "word": word,
            "freq": vocab[word]["freq"],
            "doc_freq": vocab[word]["doc_freq"],
            "tfidf_max": float(arr.max()),
            "tfidf_mean": float(arr.mean()),
            "tfidf_top_k_mean": top_k_mean,
            "word_len": len(word),
        })

    df = pd.DataFrame(rows)
    df["log_freq"] = np.log1p(df["freq"])
    df["log_doc_freq"] = np.log1p(df["doc_freq"])
    return df.sort_values("tfidf_top_k_mean", ascending=False).reset_index(drop=True)


def build_word_metrics(size: str = "small", top_k: int = 5) -> pd.DataFrame:
    import yaml  # noqa: PLC0415
    from src.tfidf.compute_tfidf import build_engine, score_all_documents  # noqa: PLC0415

    with open(ROOT / "configs/wikipedia.yaml") as f:
        wiki_cfg = yaml.safe_load(f)
    with open(ROOT / "configs/experiments.yaml") as f:
        exp_cfg = yaml.safe_load(f)

    proc = ROOT / wiki_cfg["paths"]["processed"]
    vocab_path = proc / f"vocab_{size}.json"
    docs_path = proc / f"tfidf_docs_{size}.jsonl"

    with open(vocab_path) as f:
        vocab = json.load(f)

    engine, lines = build_engine(docs_path, exp_cfg)
    word_scores = score_all_documents(engine, lines)
    df = aggregate(word_scores, vocab, top_k)

    out = proc / f"tfidf_scores_{size}.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved TF-IDF scores → {out}  ({len(df):,} words)")
    return df


if __name__ == "__main__":
    import argparse  # noqa

    p = argparse.ArgumentParser()
    p.add_argument("--size", default="small")
    args = p.parse_args()
    build_word_metrics(args.size)
