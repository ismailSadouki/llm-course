"""PROVIDED: load the three pretrained tokenizers of the assignment."""

from __future__ import annotations

PRETRAINED_MODELS = ["gpt2", "bert-base-uncased", "google/flan-t5-small"]


def load_pretrained(names=None, verbose: bool = True) -> dict:
    """Load tokenizers with AutoTokenizer. A model that cannot be loaded
    (no internet, wrong name...) is skipped with a warning, not a crash."""
    from transformers import AutoTokenizer

    loaded = {}
    for name in names or PRETRAINED_MODELS:
        try:
            loaded[name] = AutoTokenizer.from_pretrained(name)
            if verbose:
                print(f"loaded {name}: {type(loaded[name]).__name__}")
        except Exception as exc:                       # noqa: BLE001
            print(f"WARNING: could not load {name!r}: {type(exc).__name__}: {str(exc)[:120]}")
    return loaded
