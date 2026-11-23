"""PROVIDED metrics for Part D. You do not need to modify this file."""

from __future__ import annotations

import random
from collections import Counter
from typing import Iterable


def split_lines(lines: list[str], test_fraction: float = 0.2, seed: int = 0):
    """Shuffle lines reproducibly and split into (train, test)."""
    lines = list(lines)
    random.Random(seed).shuffle(lines)
    n_test = int(len(lines) * test_fraction)
    return lines[n_test:], lines[:n_test]


def words_of(texts: Iterable[str]) -> list[str]:
    return [w for t in texts for w in t.split()]


def avg_tokens_per_word(tokenizer, texts: Iterable[str]) -> float:
    """Fertility: average number of BPE tokens per whitespace-separated word
    (the end-of-word marker is part of the last token, not a separate token)."""
    n_words = n_tokens = 0
    for text in texts:
        n_words += len(text.split())
        n_tokens += len(tokenizer.encode(text))
    return n_tokens / n_words if n_words else 0.0


def compression_ratio(tokenizer, texts: Iterable[str]) -> float:
    """Characters (without spaces) per token. Higher = shorter sequences."""
    n_chars = n_tokens = 0
    for text in texts:
        n_chars += sum(len(w) for w in text.split())
        n_tokens += len(tokenizer.encode(text))
    return n_chars / n_tokens if n_tokens else 0.0


def unseen_words(train_texts: Iterable[str], test_texts: Iterable[str]) -> list[str]:
    """Distinct test words that never appear in the training texts."""
    seen = set(words_of(train_texts))
    return sorted({w for w in words_of(test_texts) if w not in seen})


def oov_rate_words(train_texts: Iterable[str], test_texts: Iterable[str]) -> float:
    """Fraction of test word occurrences whose word is absent from train."""
    seen = set(words_of(train_texts))
    test_words = words_of(test_texts)
    if not test_words:
        return 0.0
    return sum(w not in seen for w in test_words) / len(test_words)


def token_length_histogram(tokenizer) -> Counter:
    """How many vocabulary tokens have each length (in characters)."""
    return Counter(len(t.replace("</w>", "_")) for t in tokenizer.token_to_id if t != "[UNK]")
