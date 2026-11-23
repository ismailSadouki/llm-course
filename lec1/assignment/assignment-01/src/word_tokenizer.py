"""Word-level tokenizer   (Part B, task 3)

Run the tests with:   pytest tests/test_tokenizers.py -k "Word"
"""

from __future__ import annotations

import re
from collections import Counter          # useful for train()
from typing import Iterable, Optional

# ---------------------------------------------------------------------------
# PROVIDED. Tokenization convention (copy and discuss it in your report):
#   * a word is a run of letters/digits, optionally with inner apostrophes
#       "Let's" -> ["Let's"]      "L'apprentissage" -> ["L'apprentissage"]
#   * every punctuation mark is its own token
#       "models." -> ["models", "."]
#   * case is preserved
#   * whitespace is discarded  (=> decoding cannot be perfectly lossless!)
# The tests assume this convention. If you change it, justify the change and
# adapt the tests.
# ---------------------------------------------------------------------------
TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]")

NO_SPACE_BEFORE = set(".,!?;:)]}%…،؟؛")
NO_SPACE_AFTER = set("([{")


def word_tokenize(text: str) -> list[str]:
    """Split text into word and punctuation tokens."""
    return TOKEN_PATTERN.findall(text)


def detokenize(tokens: list[str]) -> str:
    """Join tokens back into text with simple spacing rules."""
    out: list[str] = []
    prev: Optional[str] = None
    for tok in tokens:
        if prev is None or tok in NO_SPACE_BEFORE or prev in NO_SPACE_AFTER:
            out.append(tok)
        else:
            out.append(" " + tok)
        prev = tok
    return "".join(out)


class WordTokenizer:
    PAD, UNK, BOS, EOS = "[PAD]", "[UNK]", "[BOS]", "[EOS]"
    SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]       # ids 0, 1, 2, 3

    # ---- PROVIDED ---------------------------------------------------------
    def __init__(self, min_freq: int = 1, max_vocab_size: Optional[int] = None) -> None:
        """
        min_freq:       tokens seen fewer times than this are left out
        max_vocab_size: max number of NON-special tokens to keep (None = all)
        """
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._reset()

    def _reset(self) -> None:
        self.token_to_id = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    # ---- TODO -------------------------------------------------------------
    def train(self, texts: Iterable[str]) -> None:
        """Build the vocabulary.   (= notebook 02, exercise 2.3)

        1. reset, then count every token produced by word_tokenize()
        2. keep tokens with count >= min_freq
        3. order them by decreasing count; break ties alphabetically
           (=> deterministic ids: frequent words get small ids)
        4. keep at most max_vocab_size of them (if it is not None)
        5. assign ids after the 4 special tokens
        """
        raise NotImplementedError

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Unknown tokens -> id of [UNK]. Optionally wrap with [BOS] ... [EOS]."""
        raise NotImplementedError

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        """Ids -> tokens -> text (use detokenize). Unknown id -> ValueError."""
        raise NotImplementedError

    def oov_rate(self, texts: Iterable[str]) -> float:
        """Fraction of token OCCURRENCES (not distinct tokens) encoded as [UNK].   (= notebook 02, exercise 2.4)

        Return 0.0 if there are no tokens at all.
        """
        raise NotImplementedError
