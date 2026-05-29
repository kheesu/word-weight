"""Download and preprocess a Wikipedia subset, saving two parallel representations:
  - processed/wikipedia/tfidf_docs.jsonl   — list of token lists for TF-IDF
  - processed/wikipedia/raw_docs.jsonl     — lightly cleaned text for tokenization
  - processed/wikipedia/vocab.json         — word → {freq, doc_freq}
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.preprocessing.clean_text import clean_for_tfidf, clean_for_tokenizer, is_valid_word


def load_config(path: str = "configs/wikipedia.yaml") -> dict:
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def stream_wikipedia(cfg: dict, n_docs: int):
    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(
        cfg["dataset_name"],
        cfg["dataset_config"],
        split=cfg["split"],
        streaming=True,
        trust_remote_code=True,
    )
    count = 0
    for example in ds:
        text = example.get("text", "")
        if len(text) < cfg["preprocessing"]["min_doc_length"]:
            continue
        yield text
        count += 1
        if count >= n_docs:
            break


def build(size: str = "small", config_path: str = "configs/wikipedia.yaml"):
    cfg = load_config(config_path)
    n_docs = cfg["sizes"][size]
    pre = cfg["preprocessing"]
    out_dir = ROOT / cfg["paths"]["processed"]
    out_dir.mkdir(parents=True, exist_ok=True)

    tfidf_path = out_dir / f"tfidf_docs_{size}.jsonl"
    raw_path = out_dir / f"raw_docs_{size}.jsonl"
    vocab_path = out_dir / f"vocab_{size}.json"

    word_freq: dict[str, int] = defaultdict(int)
    doc_freq: dict[str, int] = defaultdict(int)

    with open(tfidf_path, "w") as ft, open(raw_path, "w") as fr:
        for text in tqdm(stream_wikipedia(cfg, n_docs), total=n_docs, desc="processing"):
            tokens = clean_for_tfidf(text)
            raw = clean_for_tokenizer(text)

            valid = [
                t for t in tokens
                if is_valid_word(
                    t,
                    min_len=pre["min_word_len"],
                    max_len=pre["max_word_len"],
                    alphabetic_only=pre["alphabetic_only"],
                )
            ]

            ft.write(json.dumps(valid) + "\n")
            fr.write(json.dumps(raw) + "\n")

            seen = set(valid)
            for w in valid:
                word_freq[w] += 1
            for w in seen:
                doc_freq[w] += 1

    min_freq = pre["min_word_freq"]
    vocab = {
        w: {"freq": word_freq[w], "doc_freq": doc_freq[w]}
        for w in word_freq
        if doc_freq[w] >= min_freq
    }
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)

    print(f"Saved {n_docs} docs → {out_dir}")
    print(f"Vocabulary size (doc_freq >= {min_freq}): {len(vocab):,}")
    return tfidf_path, raw_path, vocab_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--size", default="small", choices=["small", "medium", "large"])
    args = p.parse_args()
    build(args.size)
