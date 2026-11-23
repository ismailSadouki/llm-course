"""A tiny NumPy embedding layer   (Part F)

Run the tests with:   pytest tests/test_embeddings.py
"""

from __future__ import annotations

import numpy as np


class EmbeddingMatrix:
    # ---- PROVIDED ---------------------------------------------------------
    def __init__(self, vocab_size: int, embedding_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # E in R^{V x d}, small random initial values
        self.weight = rng.normal(0.0, 0.02, size=(vocab_size, embedding_dim))

    @property
    def num_parameters(self) -> int:
        return int(self.weight.size)

    # ---- TODO -------------------------------------------------------------
    def __call__(self, token_ids) -> np.ndarray:
        """Row lookup.   (= notebook 05, exercise 5.1)

        token_ids: list / array of ints of ANY shape, e.g. (T,) or (B, T)
        returns:   array of shape token_ids.shape + (embedding_dim,)
        raises IndexError if an id is < 0 or >= vocab_size.

        Hint: NumPy "fancy indexing": self.weight[ids]
        """
        raise NotImplementedError

    def one_hot(self, token_ids) -> np.ndarray:
        """(= notebook 05, exercise 5.2)  One-hot encode: shape token_ids.shape + (vocab_size,), entries 0.0/1.0.

        Same IndexError rule as above.
        """
        raise NotImplementedError

    def lookup_via_matmul(self, token_ids) -> np.ndarray:
        """Same result as __call__, computed as one_hot(ids) @ self.weight.

        This shows that an embedding layer IS a linear layer applied to a
        one-hot vector -- lookup is just the efficient way to compute it.
        """
        raise NotImplementedError
