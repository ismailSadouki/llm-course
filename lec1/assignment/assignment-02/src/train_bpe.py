"""Train a byte-level BPE tokenizer with Hugging Face `tokenizers`   (Part E, task 7)

A `tokenizers.Tokenizer` is a PIPELINE of components:

    text -> normalizer -> pre-tokenizer -> MODEL (BPE) -> post-processor -> ids
    ids  -> DECODER -> text

For byte-level BPE you need: the BPE model, the ByteLevel pre-tokenizer, the
ByteLevel decoder, and a BpeTrainer.

You already wrote this code, step by step, in notebook 06 (exercises 6.1 – 6.3).
Here you package it as reusable functions: copy your working lines from the notebook.

Run the tests with:   pytest tests/test_bpe.py
CLI (when finished):  python -m src.train_bpe --files data/processed/train_corpus.txt --out models/bpe/tokenizer.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

DEFAULT_SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]   # they get ids 0, 1, 2, 3
PathLike = Union[str, Path]


# ---- TODO 1 -----------------------------------------------------------------
def build_bpe_tokenizer(unk_token: str = "[UNK]") -> Tokenizer:
    """Return an UNTRAINED byte-level BPE tokenizer.   (= notebook 06, exercise 6.1)

    * model:          BPE(unk_token=unk_token)
    * pre-tokenizer:  ByteLevel(add_prefix_space=False)
    * decoder:        ByteLevelDecoder()      <- without it, decode() returns 'Ã©' garbage

    (Two classes are both called ByteLevel: tokenizers.pre_tokenizers.ByteLevel and
    tokenizers.decoders.ByteLevel -- the import block above renames the decoder.)
    """
    raise NotImplementedError


# ---- TODO 2 -----------------------------------------------------------------
def make_trainer(vocab_size: int, min_frequency: int, special_tokens: Sequence[str]) -> BpeTrainer:
    """Return a BpeTrainer.   (= notebook 06, exercise 6.2)

    IMPORTANT: pass initial_alphabet=ByteLevel.alphabet(). It puts the 256 byte symbols in the
    vocabulary even if some bytes never occur in your corpus (e.g. Arabic bytes in an
    English/French corpus). That is what makes byte-level BPE unable to meet an unknown symbol.
    Also pass show_progress=False to keep the notebooks and the tests quiet.
    """
    raise NotImplementedError


# ---- TODO 3 -----------------------------------------------------------------
def train_bpe(
    files: Union[PathLike, Sequence[PathLike]],
    vocab_size: int = 2000,
    min_frequency: int = 2,
    special_tokens: Optional[Sequence[str]] = None,
) -> Tokenizer:
    """Train on one or several text files (one sentence per line).   (= notebook 06, exercise 6.3)

    * a single path is allowed: turn it into a list
    * special_tokens=None means DEFAULT_SPECIAL_TOKENS
    * the unknown token given to the model is the FIRST special token
    * tokenizer.train(files, trainer) wants a list of *strings* (not Path objects)
    * return the trained tokenizer
    """
    raise NotImplementedError


# ---- TODO 4 -----------------------------------------------------------------
def train_bpe_from_texts(
    texts: Iterable[str],
    vocab_size: int = 2000,
    min_frequency: int = 2,
    special_tokens: Optional[Sequence[str]] = None,
) -> Tokenizer:
    """Same as train_bpe but from an iterator of strings: tokenizer.train_from_iterator(...)."""
    raise NotImplementedError


# ---- PROVIDED ---------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--vocab-size", type=int, default=2000)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--out", default="models/bpe/tokenizer.json")
    args = ap.parse_args()
    tokenizer = train_bpe(args.files, args.vocab_size, args.min_frequency)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.out)
    print(f"saved {args.out} (vocab size {tokenizer.get_vocab_size()})")


if __name__ == "__main__":
    main()
