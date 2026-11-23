"""Byte Pair Encoding from scratch   (Part C -- the core of this assignment)

Character-level BPE with an end-of-word marker "</w>".

Worked example on data/toy/toy_corpus.txt ("low"x3 "lower"x2 "newer"x2 "newest"x2)

    initial word table           l o w </w>: 3      l o w e r </w>: 2
                                 n e w e r </w>: 2  n e w e s t </w>: 2

    pair counts (weighted)       (w,e)=6  (l,o)=5  (o,w)=5  (e,r)=4 ...
    merge #1  = (w, e) -> "we"   l o w </w>: 3      l o we r </w>: 2
                                 n e we r </w>: 2   n e we s t </w>: 2
    merge #2  = (l, o) -> "lo"   (count 5)

TIE-BREAKING RULE (the tests rely on it): choose the pair with the highest
frequency; if several pairs share it, choose the lexicographically smallest
pair, i.e. the smallest tuple (first_symbol, second_symbol).
Python tip:  min(pairs, key=lambda p: (-pairs[p], p))

Run the tests with:   pytest tests/test_bpe.py -x
(-x stops at the first failure: work through the classes from top to bottom.)
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Union

END_OF_WORD = "</w>"
UNK = "[UNK]"

Pair = tuple[str, str]
WordFreqs = dict[tuple[str, ...], int]   # {("l","o","w","</w>"): 3, ...}


class BPETokenizer:
    """
    Attributes you must fill in train():
        merges             ordered list of merge rules, e.g. [("w","e"), ("l","o")]
        merge_frequencies  frequency of the chosen pair at each step, e.g. [6, 5]
        merge_ranks        {pair: index in merges}   (used by encode_word)
        word_freqs         the training word table AFTER the last merge
        token_to_id / id_to_token
                           id 0 = "[UNK]", then the base symbols in sorted order
                           (characters and "</w>"), then one new token per merge,
                           in merge order.
    """

    # ---- PROVIDED ---------------------------------------------------------
    def __init__(self) -> None:
        self.merges: list[Pair] = []
        self.merge_frequencies: list[int] = []
        self.merge_ranks: dict[Pair, int] = {}
        self.word_freqs: WordFreqs = {}
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    @property
    def vocab_size(self) -> int:
        """Number of entries in token_to_id (INCLUDING [UNK])."""
        return len(self.token_to_id)

    def tokens_to_ids(self, tokens: list[str]) -> list[int]:
        unk = self.token_to_id[UNK]
        return [self.token_to_id.get(t, unk) for t in tokens]

    def ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[i] for i in ids]

    # ---- TODO 1 -----------------------------------------------------------
    def initialize_vocabulary(self, corpus: Union[str, Iterable[str]]) -> WordFreqs:
        """Corpus -> word table.   (= notebook 03, exercise 3.1)

        `corpus` is a string or an iterable of lines. Split on whitespace,
        count the words, and represent each word as a TUPLE of its characters
        followed by END_OF_WORD:

            "low" (seen 3 times) -> {("l", "o", "w", "</w>"): 3}

        Hint: Counter, and tuple(word) + (END_OF_WORD,).
        """
        raise NotImplementedError

    # ---- TODO 2 -----------------------------------------------------------
    def get_pair_frequencies(self, word_freqs: WordFreqs) -> Counter:
        """Count adjacent symbol pairs, WEIGHTED by word frequency.   (= notebook 03, exercise 3.2)

        {("l","o","w","</w>"): 3}  ->  (l,o):3  (o,w):3  (w,</w>):3

        A pair that occurs twice inside one word counts twice.
        Hint: zip(symbols, symbols[1:])
        """
        raise NotImplementedError

    # ---- TODO 3 -----------------------------------------------------------
    def merge_pair(self, pair: Pair, word_freqs: WordFreqs) -> WordFreqs:
        """(= notebook 03, exercise 3.4)  Return a NEW word table in which every occurrence of `pair` is
        replaced by the concatenated symbol.

            merge_pair(("l","o"), {("l","o","w","</w>"): 3}) -> {("lo","w","</w>"): 3}

        Requirements:
          * do not mutate the input
          * scan each word left to right:  ("a","a","a") merged on ("a","a")
            gives ("aa","a")  -- not ("a","aa")
          * total frequency is preserved
          * if two words become identical, add their frequencies
        """
        raise NotImplementedError

    # ---- TODO 4 -----------------------------------------------------------
    def train(self, corpus: Union[str, Iterable[str]], num_merges: int = 10) -> "BPETokenizer":
        """Learn up to `num_merges` merge rules. Return self.   (= notebook 03, exercises 3.3 + 3.5)

        Loop:
          1. pairs = get_pair_frequencies(word_table)
          2. if there are no pairs left: stop early
          3. best = most frequent pair (tie rule in the module docstring)
          4. record it in merges / merge_frequencies
          5. word_table = merge_pair(best, word_table)

        After the loop fill: word_freqs, merge_ranks, token_to_id, id_to_token.
        Calling train() twice must start from scratch (reset merges etc.).
        """
        raise NotImplementedError

    # ---- TODO 5 -----------------------------------------------------------
    def encode_word(self, word: str) -> list[str]:
        """Segment ONE word (no spaces) with the learned merges.   (= notebook 03, exercise 3.6)

        Start from list(word) + ["</w>"], then apply the merge rules IN THE
        ORDER THEY WERE LEARNED. Two valid strategies:

          A) for each merge in self.merges: scan the word and merge it
          B) repeat: among the adjacent pairs of the word that are in
             merge_ranks, take the one with the SMALLEST rank, merge all its
             occurrences; stop when no pair is mergeable.
             (B is faster for big vocabularies -- you will need it on real data.)

        Characters never seen in training stay as single symbols (this is the
        weakness that byte-level BPE fixes, see question C.5-8).
        """
        raise NotImplementedError

    def encode(self, text: str) -> list[str]:
        """Tokenize a sentence: split on whitespace, encode_word each word,
        concatenate the token lists."""
        raise NotImplementedError

    def decode(self, tokens: list[str]) -> str:
        """Inverse of encode: concatenate, turn "</w>" into a space, strip.

        decode(encode("low lower")) == "low lower"
        (runs of whitespace in the input are normalized to one space)
        """
        raise NotImplementedError
