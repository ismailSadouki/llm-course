"""Tests that need the real pretrained tokenizers (internet access to the Hugging Face Hub).

They are SKIPPED automatically when the models cannot be downloaded.
"""
import pytest

from src.pretrained import PRETRAINED_MODELS, load_pretrained


@pytest.fixture(scope="module")
def pretrained():
    loaded = load_pretrained(verbose=False)
    if len(loaded) < len(PRETRAINED_MODELS):
        pytest.skip("pretrained tokenizers are not available (no internet / Hub unreachable)")
    return loaded


def test_all_three_are_fast_tokenizers(pretrained):
    assert all(tok.is_fast for tok in pretrained.values())


def test_known_vocabulary_sizes(pretrained):
    assert pretrained["gpt2"].vocab_size == 50257
    assert pretrained["bert-base-uncased"].vocab_size == 30522


def test_gpt2_is_lossless_on_any_text(pretrained):
    tok = pretrained["gpt2"]
    for text in ["Café ça va ?", "الذكاء الاصطناعي", "日本語 🙂", "  two  spaces "]:
        assert tok.decode(tok.encode(text)) == text


def test_bert_uncased_lowercases_and_strips_accents(pretrained):
    tok = pretrained["bert-base-uncased"]
    assert tok.tokenize("HELLO World") == ["hello", "world"]
    assert tok.tokenize("Café") == tok.tokenize("cafe")


def test_bert_adds_cls_and_sep_by_default(pretrained):
    tok = pretrained["bert-base-uncased"]
    ids = tok.encode("hello")
    assert ids[0] == tok.cls_token_id and ids[-1] == tok.sep_token_id


def test_t5_decode_of_plain_english_is_readable(pretrained):
    tok = pretrained["google/flan-t5-small"]
    assert tok.decode(tok.encode("Hello, how are you?", add_special_tokens=False)) == "Hello, how are you?"


def test_arabic_costs_more_than_english_for_every_pretrained_tokenizer(pretrained):
    en = "Artificial intelligence is changing the world very fast."
    ar = "الذكاء الاصطناعي يتغير العالم بسرعة كبيرة جدا."
    for name, tok in pretrained.items():
        tpw_en = len(tok.tokenize(en)) / len(en.split())
        tpw_ar = len(tok.tokenize(ar)) / len(ar.split())
        assert tpw_ar > tpw_en, name
