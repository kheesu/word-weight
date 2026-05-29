"""Compute per-word tokenization costs for all tokenizers in the vocabulary."""

import json
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.tokenization.load_tokenizers import TokenizerWrapper


def word_token_cost(word: str, tok: TokenizerWrapper) -> dict:
    ids = tok.encode(word)
    n = len(ids)
    length = len(word)
    return {
        "word": word,
        "tokenizer": tok.name,
        "token_count": n,
        "tokens_per_char": n / length if length else 0.0,
        "chars_per_token": length / n if n else 0.0,
        "is_split": int(n > 1),
    }


def compute_for_vocab(
    vocab: Iterable[str],
    tokenizers: list[TokenizerWrapper],
    out_path: Path | None = None,
) -> pd.DataFrame:
    rows = []
    words = list(vocab)
    for tok in tokenizers:
        for word in tqdm(words, desc=f"encoding [{tok.name}]"):
            rows.append(word_token_cost(word, tok))

    df = pd.DataFrame(rows)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"Saved token costs → {out_path}")
    return df


def load_vocab_words(vocab_path: Path, min_doc_freq: int = 3) -> list[str]:
    with open(vocab_path) as f:
        vocab = json.load(f)
    return [w for w, stats in vocab.items() if stats["doc_freq"] >= min_doc_freq]


if __name__ == "__main__":
    import argparse, yaml  # noqa: E401

    p = argparse.ArgumentParser()
    p.add_argument("--size", default="small")
    args = p.parse_args()

    from src.tokenization.load_tokenizers import load_custom_bpe, load_gpt2, load_bert  # noqa

    with open(ROOT / "configs/wikipedia.yaml") as f:
        wiki_cfg = yaml.safe_load(f)
    with open(ROOT / "configs/tokenizers.yaml") as f:
        tok_cfg = yaml.safe_load(f)

    vocab_path = ROOT / wiki_cfg["paths"]["processed"] / f"vocab_{args.size}.json"
    words = load_vocab_words(vocab_path)

    vs = tok_cfg["custom_bpe"]["default_vocab_size"]
    bpe_path = ROOT / tok_cfg["custom_bpe"]["model_dir"] / f"bpe_{args.size}_{vs}.json"
    tokenizers = [load_custom_bpe(str(bpe_path)), load_gpt2(), load_bert()]

    out = ROOT / "data/processed" / f"token_costs_{args.size}.parquet"
    compute_for_vocab(words, tokenizers, out)
