"""Train a custom BPE model using rusty_bpe on the processed corpus."""

import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_configs():
    with open(ROOT / "configs/wikipedia.yaml") as f:
        wiki = yaml.safe_load(f)
    with open(ROOT / "configs/tokenizers.yaml") as f:
        tok = yaml.safe_load(f)
    return wiki, tok


def train(size: str = "small", vocab_size: int | None = None):
    import rusty_bpe  # noqa: PLC0415

    wiki_cfg, tok_cfg = load_configs()
    bpe_cfg = tok_cfg["custom_bpe"]
    vs = vocab_size or bpe_cfg["default_vocab_size"]

    raw_path = ROOT / wiki_cfg["paths"]["processed"] / f"raw_docs_{size}.jsonl"
    model_dir = ROOT / bpe_cfg["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"bpe_{size}_{vs}.json"

    if model_path.exists():
        print(f"Model already exists: {model_path}")
        return str(model_path)

    print(f"Loading raw docs from {raw_path} ...")
    texts = []
    with open(raw_path) as f:
        for line in tqdm(f, desc="loading"):
            texts.append(json.loads(line))

    corpus = " ".join(texts)
    print(f"Training BPE (vocab_size={vs}) on {len(corpus):,} chars ...")
    vocab = rusty_bpe.train(corpus, vs, bpe_cfg["min_freq"])
    vocab.save(str(model_path))
    print(f"Saved BPE model → {model_path}  (vocab_size={vocab.vocab_size})")
    return str(model_path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--size", default="small")
    p.add_argument("--vocab-size", type=int, default=None)
    args = p.parse_args()
    train(args.size, args.vocab_size)
