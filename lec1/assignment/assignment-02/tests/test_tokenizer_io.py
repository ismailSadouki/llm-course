import json

import pytest
from tokenizers import Tokenizer

from src.tokenizer_io import find_mismatches, load_tokenizer, save_tokenizer, to_transformers


@pytest.fixture(params=["bpe", "wordpiece"])
def trained(request, bpe, wordpiece):
    return {"bpe": bpe, "wordpiece": wordpiece}[request.param]


class TestSaveAndLoad:
    def test_save_creates_file_and_parent_folders(self, trained, tmp_path):
        path = save_tokenizer(trained, tmp_path / "models" / "x" / "tokenizer.json")
        assert path.exists() and path.name == "tokenizer.json"

    def test_saved_file_is_valid_json_with_all_components(self, trained, tmp_path):
        path = save_tokenizer(trained, tmp_path / "tokenizer.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        # the vocabulary alone is NOT the tokenizer: the whole pipeline is stored
        for key in ["model", "pre_tokenizer", "decoder", "added_tokens"]:
            assert key in data
        assert data["model"]["type"] in {"BPE", "WordPiece"}
        assert "vocab" in data["model"]

    def test_load_returns_a_tokenizer(self, trained, tmp_path):
        path = save_tokenizer(trained, tmp_path / "tokenizer.json")
        assert isinstance(load_tokenizer(path), Tokenizer)

    def test_encoding_is_identical_after_reloading(self, trained, tmp_path, sentences):
        reloaded = load_tokenizer(save_tokenizer(trained, tmp_path / "tokenizer.json"))
        assert len(sentences) == 20
        for s in sentences:
            assert trained.encode(s).ids == reloaded.encode(s).ids
            assert trained.encode(s).tokens == reloaded.encode(s).tokens

    def test_decoding_is_identical_after_reloading(self, trained, tmp_path, sentences):
        reloaded = load_tokenizer(save_tokenizer(trained, tmp_path / "tokenizer.json"))
        for s in sentences:
            ids = trained.encode(s).ids
            assert trained.decode(ids) == reloaded.decode(ids)

    def test_special_tokens_survive_reloading(self, trained, tmp_path):
        reloaded = load_tokenizer(save_tokenizer(trained, tmp_path / "tokenizer.json"))
        assert reloaded.token_to_id("[UNK]") == 0
        assert reloaded.token_to_id("[PAD]") == 1

    def test_vocab_size_survives_reloading(self, trained, tmp_path):
        reloaded = load_tokenizer(save_tokenizer(trained, tmp_path / "tokenizer.json"))
        assert reloaded.get_vocab_size() == trained.get_vocab_size()


class TestFindMismatches:
    def test_no_mismatch_for_the_same_tokenizer(self, bpe, sentences):
        assert find_mismatches(bpe, bpe, sentences) == []

    def test_no_mismatch_after_reloading(self, bpe, tmp_path, sentences):
        reloaded = load_tokenizer(save_tokenizer(bpe, tmp_path / "t.json"))
        assert find_mismatches(bpe, reloaded, sentences) == []

    def test_detects_different_tokenizers(self, bpe, wordpiece, sentences):
        found = find_mismatches(bpe, wordpiece, sentences)
        assert len(found) == len(sentences)
        assert set(found[0]) == {"sentence", "ids_before", "ids_after", "decoded_before", "decoded_after"}

    def test_reports_the_offending_sentence(self, bpe, wordpiece):
        found = find_mismatches(bpe, wordpiece, ["Hello"])
        assert found[0]["sentence"] == "Hello"


class TestTransformersWrapper:
    @pytest.fixture
    def fast(self, bpe, tmp_path):
        path = save_tokenizer(bpe, tmp_path / "tokenizer.json")
        return to_transformers(path, unk_token="[UNK]", pad_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]")

    def test_is_a_fast_tokenizer(self, fast):
        assert fast.is_fast

    def test_special_tokens_are_declared(self, fast):
        assert fast.unk_token == "[UNK]" and fast.unk_token_id == 0
        assert fast.pad_token == "[PAD]" and fast.pad_token_id == 1
        assert fast.bos_token_id == 2 and fast.eos_token_id == 3
        assert {"[UNK]", "[PAD]", "[BOS]", "[EOS]"} <= set(fast.all_special_tokens)

    def test_only_declared_tokens_are_set(self, bpe, tmp_path):
        fast = to_transformers(save_tokenizer(bpe, tmp_path / "t.json"), pad_token=None, unk_token="[UNK]")
        assert fast.pad_token is None

    def test_same_ids_as_the_raw_tokenizer(self, fast, bpe, sentences):
        for s in sentences:
            assert fast.encode(s, add_special_tokens=False) == bpe.encode(s).ids

    def test_same_tokens_as_the_raw_tokenizer(self, fast, bpe):
        assert fast.tokenize("Café au lait") == bpe.encode("Café au lait").tokens

    def test_decode_roundtrip(self, fast, sentences):
        for s in sentences:
            assert fast.decode(fast.encode(s, add_special_tokens=False)) == s

    def test_vocab_size_and_len(self, fast):
        assert fast.vocab_size == 2000 and len(fast) == 2000

    def test_padding_and_attention_mask(self, fast):
        batch = fast(["a b", "a b c d e f"], padding=True)
        assert len(batch["input_ids"][0]) == len(batch["input_ids"][1])
        assert batch["attention_mask"][0][-1] == 0
        assert batch["input_ids"][0][-1] == fast.pad_token_id

    def test_save_pretrained_and_autotokenizer(self, fast, tmp_path, sentences):
        from transformers import AutoTokenizer

        folder = tmp_path / "hf_tokenizer"
        fast.save_pretrained(str(folder))
        assert (folder / "tokenizer.json").exists()
        again = AutoTokenizer.from_pretrained(str(folder))
        for s in sentences:
            assert again.encode(s, add_special_tokens=False) == fast.encode(s, add_special_tokens=False)
        assert again.pad_token == "[PAD]"
