"""Load all tokenizers as a uniform interface: encode(word) → list[int]."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@dataclass
class TokenizerWrapper:
    name: str
    encode: Callable[[str], list[int]]
    vocab_size: int
    _meta: dict = field(default_factory=dict)


def load_custom_bpe(model_path: str) -> TokenizerWrapper:
    import rusty_bpe  # noqa: PLC0415

    tok = rusty_bpe.Tokenizer(model_path)
    return TokenizerWrapper(
        name="custom_bpe",
        encode=tok.encode,
        vocab_size=tok.vocab_size,
    )


def load_gpt2() -> TokenizerWrapper:
    import tiktoken  # noqa: PLC0415

    enc = tiktoken.get_encoding("gpt2")
    return TokenizerWrapper(
        name="gpt2",
        encode=enc.encode,
        vocab_size=enc.n_vocab,
    )


def load_bert() -> TokenizerWrapper:
    from transformers import BertTokenizer  # noqa: PLC0415

    tok = BertTokenizer.from_pretrained("bert-base-uncased")

    def encode(word: str) -> list[int]:
        return tok.encode(word, add_special_tokens=False)

    return TokenizerWrapper(
        name="bert_wordpiece",
        encode=encode,
        vocab_size=tok.vocab_size,
    )


def load_sentencepiece(model_path: str | None = None, train_sentences: list[str] | None = None,
                       vocab_size: int = 32000) -> TokenizerWrapper:
    import sentencepiece as spm  # noqa: PLC0415

    sp = spm.SentencePieceProcessor()

    if model_path and Path(model_path).exists():
        sp.load(model_path)
    elif train_sentences:
        import tempfile, os  # noqa: PLC0415, E401

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            for sentence in train_sentences:
                tmp.write(sentence.replace("\n", " ") + "\n")
            tmp_path = tmp.name
        out_prefix = str(ROOT / "data/processed/sentencepiece_model")
        Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
        spm.SentencePieceTrainer.train(
            input=tmp_path,
            model_prefix=out_prefix,
            vocab_size=vocab_size,
            model_type="unigram",
            character_coverage=0.9995,
            max_sentence_length=1_000_000,
        )
        os.unlink(tmp_path)
        sp.load(out_prefix + ".model")
    else:
        raise ValueError("Either model_path or train_sentences must be provided")

    return TokenizerWrapper(
        name="sentencepiece_unigram",
        encode=lambda w: sp.encode(w, out_type=int),
        vocab_size=sp.get_piece_size(),
    )


def load_all(bpe_model_path: str, sp_model_path: str | None = None,
             sp_train_sentences: list[str] | None = None) -> list[TokenizerWrapper]:
    tokenizers = [
        load_custom_bpe(bpe_model_path),
        load_gpt2(),
        load_bert(),
    ]
    try:
        tokenizers.append(
            load_sentencepiece(sp_model_path, sp_train_sentences)
        )
    except Exception as e:
        print(f"[warn] SentencePiece skipped: {e}")
    return tokenizers
