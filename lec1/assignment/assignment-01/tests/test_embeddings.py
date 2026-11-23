import numpy as np
import pytest

from src.embeddings import EmbeddingMatrix


@pytest.fixture
def emb():
    return EmbeddingMatrix(vocab_size=10_000, embedding_dim=256, seed=0)


def test_weight_shape(emb):
    assert emb.weight.shape == (10_000, 256)


def test_number_of_parameters(emb):
    assert emb.num_parameters == 10_000 * 256 == 2_560_000


def test_sequence_lookup_shape(emb):
    ids = list(range(12))                       # a sentence of 12 tokens
    assert emb(ids).shape == (12, 256)


def test_batch_lookup_shape(emb):
    ids = [[12, 45, 78, 91, 103]]               # (batch=1, seq=5)
    assert emb(ids).shape == (1, 5, 256)


def test_lookup_returns_the_matching_row(emb):
    out = emb([7, 42])
    np.testing.assert_array_equal(out[0], emb.weight[7])
    np.testing.assert_array_equal(out[1], emb.weight[42])


def test_same_token_same_vector(emb):
    out = emb([5, 9, 5])
    np.testing.assert_array_equal(out[0], out[2])


def test_out_of_range_ids_raise(emb):
    with pytest.raises(IndexError):
        emb([10_000])
    with pytest.raises(IndexError):
        emb([-1])


def test_one_hot_shape_and_content():
    e = EmbeddingMatrix(6, 3)
    oh = e.one_hot([1, 4])
    assert oh.shape == (2, 6)
    assert oh.sum() == 2
    assert oh[0, 1] == 1 and oh[1, 4] == 1


def test_lookup_equals_one_hot_matmul():
    e = EmbeddingMatrix(50, 8, seed=3)
    ids = [[1, 2, 3], [4, 5, 49]]
    np.testing.assert_allclose(e.lookup_via_matmul(ids), e(ids))


def test_seed_makes_initialization_reproducible():
    a = EmbeddingMatrix(5, 4, seed=1).weight
    b = EmbeddingMatrix(5, 4, seed=1).weight
    c = EmbeddingMatrix(5, 4, seed=2).weight
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
