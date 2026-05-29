"""Build a TF-IDF engine from the processed corpus using rusty_tfidf."""

import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def build_engine(tfidf_docs_path: Path, cfg: dict):
    import rusty_tfidf  # noqa: PLC0415

    tf_method = getattr(rusty_tfidf.TfMethod, cfg["tfidf"]["tf_method"])
    idf_method = getattr(rusty_tfidf.IdfMethod, cfg["tfidf"]["idf_method"])
    engine = rusty_tfidf.TfIdf(tf_method, idf_method)

    with open(tfidf_docs_path) as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="building IDF index"):
        tokens = json.loads(line)
        engine.add_document(tokens)

    return engine, lines


def score_all_documents(engine, lines: list[str]) -> dict[str, list[float]]:
    """Return {word: [tfidf_score, ...]} over all docs where the word appears."""
    word_scores: dict[str, list[float]] = {}
    for line in tqdm(lines, desc="scoring documents"):
        tokens = json.loads(line)
        scores = engine.scores_for_document(tokens)
        for word, score in scores.items():
            if score > 0:
                word_scores.setdefault(word, []).append(score)
    return word_scores


if __name__ == "__main__":
    import argparse  # noqa

    p = argparse.ArgumentParser()
    p.add_argument("--size", default="small")
    args = p.parse_args()

    with open(ROOT / "configs/wikipedia.yaml") as f:
        wiki_cfg = yaml.safe_load(f)
    with open(ROOT / "configs/experiments.yaml") as f:
        exp_cfg = yaml.safe_load(f)

    docs_path = ROOT / wiki_cfg["paths"]["processed"] / f"tfidf_docs_{args.size}.jsonl"
    engine, lines = build_engine(docs_path, exp_cfg)
    print(f"Corpus: {engine.doc_count} docs, {engine.vocabulary_size} unique terms")
