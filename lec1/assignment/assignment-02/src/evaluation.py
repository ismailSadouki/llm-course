"""Tokenizer benchmark: adapter, per-sentence results, summary metrics   (Part G)

Warm-up exercises in notebook 07: 7.1 (count tokens through both APIs -> TokenizerAdapter) and
7.2 (compute the metrics by hand with pandas -> summarize).

Run the tests with:   pytest tests/test_evaluation.py

Two difficulties you must handle (the tests check both):
  1. `tokenizers.Tokenizer` and Transformers tokenizers have DIFFERENT APIs.
  2. Fair comparison: never count the special tokens a tokenizer adds on its own
     ([CLS] [SEP] for BERT, </s> for T5...). Use add_special_tokens=False everywhere.
"""

from __future__ import annotations

import time
from typing import Iterable, Optional, Sequence

import pandas as pd
from tokenizers import Tokenizer


# ---- TODO 1 -----------------------------------------------------------------
class TokenizerAdapter:
    """Give `tokenizers.Tokenizer` and Transformers tokenizers the same interface."""

    def __init__(self, name: str, tokenizer, unk_token: Optional[str] = None) -> None:
        """PROVIDED skeleton -- complete the unknown-token detection (marked below)."""
        self.name = name
        self.tokenizer = tokenizer
        self._is_raw = isinstance(tokenizer, Tokenizer)
        # TODO: if unk_token is None, find it yourself:
        #   * raw Tokenizer          -> tokenizer.model.unk_token   (may be None)
        #   * Transformers tokenizer -> tokenizer.unk_token
        self.unk_token = unk_token

    @property
    def vocab_size(self) -> int:
        """Size of the vocabulary INCLUDING added/special tokens.

        raw:          tokenizer.get_vocab_size(with_added_tokens=True)
        Transformers: len(tokenizer)      (NOT tokenizer.vocab_size, see notebook 01)
        """
        raise NotImplementedError

    @property
    def unk_id(self) -> Optional[int]:
        """Id of the unknown token, or None if the tokenizer has none.

        raw:          tokenizer.token_to_id(token)
        Transformers: tokenizer.convert_tokens_to_ids(token)
        """
        raise NotImplementedError

    def tokenize(self, text: str) -> list[str]:
        """Token strings WITHOUT special tokens.

        raw:          tokenizer.encode(text, add_special_tokens=False).tokens
        Transformers: tokenizer.tokenize(text)
        """
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """Token ids WITHOUT special tokens (add_special_tokens=False)."""
        raise NotImplementedError


RESULT_COLUMNS = ["tokenizer", "language", "sentence", "n_chars", "n_bytes", "n_words", "n_tokens", "n_unk"]


# ---- TODO 2 -----------------------------------------------------------------
def evaluate_sentences(adapter: TokenizerAdapter, df: pd.DataFrame) -> pd.DataFrame:
    """One row per sentence of `df` (columns "language" and "text").

    Output columns = RESULT_COLUMNS, in this order:
        tokenizer   adapter.name
        language    from df
        sentence    from df
        n_chars     len(sentence)                      (Unicode code points)
        n_bytes     len(sentence.encode("utf-8"))
        n_words     number of whitespace-separated words
        n_tokens    number of tokens
        n_unk       number of tokens whose id == adapter.unk_id  (0 if unk_id is None)
    """
    raise NotImplementedError


# ---- TODO 3 -----------------------------------------------------------------
def summarize(
    results: pd.DataFrame,
    group_cols: Sequence[str] = ("tokenizer",),
    vocab_sizes: Optional[dict] = None,
) -> pd.DataFrame:
    """Aggregate sentence-level results: one row per group (groupby with sort=False).

    Output = the group columns, then exactly these columns:
        avg_tokens        mean of n_tokens per sentence
        max_tokens        max of n_tokens
        chars_per_token   sum(n_chars) / sum(n_tokens)      <- ratio of TOTALS, not mean of ratios
        bytes_per_token   sum(n_bytes) / sum(n_tokens)
        tokens_per_word   sum(n_tokens) / sum(n_words)      (also called fertility)
        unk_rate          sum(n_unk) / sum(n_tokens)
        n_sentences       number of sentences in the group
        vocab_size        ONLY if `vocab_sizes` ({tokenizer name: size}) is given and
                          "tokenizer" is one of the group columns

    Why chars_per_token and not the average of per-sentence ratios? Try both on the tests'
    tiny example and see why long sentences must weigh more than short ones.
    """
    raise NotImplementedError


# ---- PROVIDED ---------------------------------------------------------------
def time_encoding(adapter: TokenizerAdapter, sentences: Iterable[str], repeats: int = 5, warmup: int = 1) -> float:
    """Mean wall-clock seconds to tokenize ALL `sentences` once (sentence by sentence).

    One untimed warm-up pass, then `repeats` timed passes; returns the mean.
    """
    sentences = list(sentences)
    for _ in range(warmup):
        for s in sentences:
            adapter.encode(s)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for s in sentences:
            adapter.encode(s)
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def parity(summary_by_language: pd.DataFrame, reference: str = "en", metric: str = "tokens_per_word") -> pd.DataFrame:
    """Per tokenizer: metric(language) / metric(reference). 1.0 = same cost as the reference.

    Input: output of summarize(results, group_cols=["tokenizer", "language"]).
    """
    pivot = summary_by_language.pivot(index="tokenizer", columns="language", values=metric)
    return pivot.div(pivot[reference], axis=0)
