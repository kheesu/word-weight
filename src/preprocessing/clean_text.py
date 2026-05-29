"""Text cleaning utilities for TF-IDF and tokenizer evaluation."""

import re
import unicodedata


_MARKUP_RE = re.compile(r"<[^>]+>|\{\{[^}]*\}\}|\[\[(?:[^|\]]*\|)?([^\]]*)\]\]")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def strip_markup(text: str) -> str:
    """Remove HTML tags and common wiki markup."""
    text = _MARKUP_RE.sub(lambda m: m.group(1) or "", text)
    return text


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def clean_for_tfidf(text: str) -> list[str]:
    """Return lowercase alphabetic tokens suitable for TF-IDF.

    Uses rusty_tfidf.tokenize which splits on non-alphanumeric chars and lowercases.
    """
    import rusty_tfidf  # noqa: PLC0415

    text = strip_markup(text)
    text = normalize_unicode(text)
    return rusty_tfidf.tokenize(text)


def clean_for_tokenizer(text: str) -> str:
    """Light cleaning for tokenizer evaluation — preserve casing and spacing."""
    text = strip_markup(text)
    text = normalize_unicode(text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def is_valid_word(
    word: str,
    min_len: int = 2,
    max_len: int = 40,
    alphabetic_only: bool = True,
) -> bool:
    if len(word) < min_len or len(word) > max_len:
        return False
    if alphabetic_only and not word.isalpha():
        return False
    return True
