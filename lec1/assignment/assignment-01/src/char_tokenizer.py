"""Character-level tokenizer   (Part B, task 2)

Run the tests with:   pytest tests/test_tokenizers.py -k Character
"""

from __future__ import annotations

from typing import Iterable


class CharacterTokenizer:
    PAD, UNK, BOS, EOS = "[PAD]", "[UNK]", "[BOS]", "[EOS]"
    SPECIAL_TOKENS = [PAD, UNK, BOS, EOS]       # they get ids 0, 1, 2, 3

    # ---- PROVIDED ---------------------------------------------------------
    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self._reset()

    def _reset(self) -> None:
        """Forget everything except the special tokens."""
        self.token_to_id = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    # ---- TODO -------------------------------------------------------------
    def train(self, texts: Iterable[str]) -> None:
        """Build the vocabulary.   (= notebook 02, exercise 2.1)

        * start from a clean state (self._reset())
        * special tokens keep ids 0..3
        * then every DISTINCT character of `texts`, in sorted order
          (sorted => the ids are reproducible from run to run)
        * keep token_to_id and id_to_token consistent
        """
        raise NotImplementedError

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """One id per character; unknown characters -> id of [UNK].   (= notebook 02, exercise 2.2)

        If add_special_tokens is True, wrap the sequence: [BOS] ... [EOS].
        """
        raise NotImplementedError

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        """Inverse of encode.

        * skip_special_tokens=True  -> drop [PAD] [UNK] [BOS] [EOS]
        * skip_special_tokens=False -> keep them, written as "[BOS]" etc.
        * an id that is not in the vocabulary -> raise ValueError
        """
        raise NotImplementedError
