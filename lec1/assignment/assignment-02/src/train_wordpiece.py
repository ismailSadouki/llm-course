"""Train a WordPiece tokenizer (BERT style) with Hugging Face `tokenizers`   (Part E, task 8)

Pipeline:  text -> BertNormalizer -> BertPreTokenizer -> WordPiece model -> ids
           ids  -> WordPiece decoder (glues '##' pieces back) -> text

You already wrote this code, step by step, in notebook 06 (exercises 6.4 – 6.5).
Here you package it as reusable functions.

Run the tests with:   pytest tests/test_wordpiece.py
CLI (when finished):  python -m src.train_wordpiece --files data/processed/train_corpus.txt --out models/wordpiece/tokenizer.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.trainers import WordPieceTrainer

DEFAULT_SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]   # ids 0..4
PathLike = Union[str, Path]


# ---- TODO 1 -----------------------------------------------------------------
def build_wordpiece_tokenizer(
    lowercase: bool = True, strip_accents: bool = True, unk_token: str = "[UNK]"
) -> Tokenizer:
    """Return an UNTRAINED WordPiece tokenizer with all four components:   (= notebook 06, exercise 6.4)

    * model:          WordPiece(unk_token=unk_token)
    * normalizer:     BertNormalizer(lowercase=lowercase, strip_accents=strip_accents)
    * pre-tokenizer:  BertPreTokenizer()
    * decoder:        WordPieceDecoder(prefix="##")
    """
    raise NotImplementedError


# ---- TODO 2 -----------------------------------------------------------------
def make_trainer(vocab_size: int, min_frequency: int, special_tokens: Sequence[str]) -> WordPieceTrainer:
    """WordPieceTrainer with continuing_subword_prefix="##" and show_progress=False.   (= notebook 06, exercise 6.5)"""
    raise NotImplementedError


# ---- TODO 3 -----------------------------------------------------------------
def train_wordpiece(
    files: Union[PathLike, Sequence[PathLike]],
    vocab_size: int = 2000,
    min_frequency: int = 2,
    special_tokens: Optional[Sequence[str]] = None,
    lowercase: bool = True,
    strip_accents: bool = True,
) -> Tokenizer:
    """Train on text files. Same conventions as train_bpe (first special token = unknown token)."""
    raise NotImplementedError


# ---- TODO 4 -----------------------------------------------------------------
def train_wordpiece_from_texts(
    texts: Iterable[str],
    vocab_size: int = 2000,
    min_frequency: int = 2,
    special_tokens: Optional[Sequence[str]] = None,
    lowercase: bool = True,
    strip_accents: bool = True,
) -> Tokenizer:
    """Same as train_wordpiece but from an iterator of strings."""
    raise NotImplementedError


# ---- PROVIDED ---------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--vocab-size", type=int, default=2000)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--out", default="models/wordpiece/tokenizer.json")
    args = ap.parse_args()
    tokenizer = train_wordpiece(args.files, args.vocab_size, args.min_frequency)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.out)
    print(f"saved {args.out} (vocab size {tokenizer.get_vocab_size()})")


if __name__ == "__main__":
    main()
