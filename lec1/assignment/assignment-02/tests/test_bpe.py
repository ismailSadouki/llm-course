import pytest
from tokenizers.pre_tokenizers import ByteLevel

from src.train_bpe import (
    DEFAULT_SPECIAL_TOKENS,
    build_bpe_tokenizer,
    train_bpe,
    train_bpe_from_texts,
)

TEXTS = [
    "Hello, how are you?",
    "Café crème à la française",
    "الذكاء الاصطناعي يتطور بسرعة",
    "日本語のテキスト 🙂 emoji",
    "naïve façade — “quotes” and 123,456.78",
]


class TestUntrainedTokenizer:
    def test_components(self):
        tok = build_bpe_tokenizer()
        assert type(tok.pre_tokenizer).__name__ == "ByteLevel"
        assert type(tok.decoder).__name__ == "ByteLevel"
        assert type(tok.model).__name__ == "BPE"

    def test_unk_token_is_configured(self):
        assert build_bpe_tokenizer(unk_token="<unk>").model.unk_token == "<unk>"


class TestTraining:
    def test_default_special_tokens_and_ids(self, bpe):
        assert DEFAULT_SPECIAL_TOKENS == ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
        assert [bpe.token_to_id(t) for t in DEFAULT_SPECIAL_TOKENS] == [0, 1, 2, 3]

    def test_requested_vocab_size_is_reached_on_a_big_corpus(self, bpe):
        assert bpe.get_vocab_size() == 2000

    def test_vocab_size_can_be_smaller_than_requested(self, toy_file):
        tok = train_bpe(toy_file, vocab_size=5000, min_frequency=2)
        assert tok.get_vocab_size() < 5000          # the corpus ran out of frequent pairs

    def test_vocabulary_contains_the_256_byte_symbols(self, bpe):
        assert set(ByteLevel.alphabet()) <= set(bpe.get_vocab())
        assert len(ByteLevel.alphabet()) == 256

    def test_ids_are_unique_and_contiguous(self, bpe):
        ids = sorted(bpe.get_vocab().values())
        assert ids == list(range(len(ids)))

    def test_frequent_words_become_single_tokens(self, bpe):
        assert bpe.encode(" the").tokens == ["Ġthe"]
        assert len(bpe.encode(" parlement").tokens) <= 2

    def test_min_frequency_shrinks_the_vocabulary_of_a_small_corpus(self, toy_file):
        low = train_bpe(toy_file, vocab_size=5000, min_frequency=1).get_vocab_size()
        high = train_bpe(toy_file, vocab_size=5000, min_frequency=6).get_vocab_size()
        assert high < low

    def test_training_is_deterministic(self, toy_file):
        a = train_bpe(toy_file, vocab_size=400).get_vocab()
        b = train_bpe(toy_file, vocab_size=400).get_vocab()
        assert a == b

    def test_custom_special_tokens(self, toy_file):
        tok = train_bpe(toy_file, vocab_size=400, special_tokens=["<unk>", "<pad>"])
        assert tok.token_to_id("<unk>") == 0 and tok.token_to_id("<pad>") == 1
        assert tok.token_to_id("[BOS]") is None

    def test_accepts_a_list_of_files(self, toy_file):
        tok = train_bpe([toy_file, toy_file], vocab_size=400)
        assert tok.get_vocab_size() > 256

    def test_train_from_texts_matches_train_from_file(self, toy_file):
        lines = toy_file.read_text(encoding="utf-8").splitlines()
        a = train_bpe(toy_file, vocab_size=400).get_vocab()
        b = train_bpe_from_texts(lines, vocab_size=400).get_vocab()
        assert a == b


class TestEncodingBehaviour:
    @pytest.mark.parametrize("text", TEXTS)
    def test_lossless_roundtrip(self, bpe, text):
        assert bpe.decode(bpe.encode(text).ids) == text

    @pytest.mark.parametrize("text", TEXTS)
    def test_never_produces_unknown_token(self, bpe, text):
        assert 0 not in bpe.encode(text).ids

    def test_special_tokens_are_atomic(self, bpe):
        enc = bpe.encode("[BOS] hello [EOS]")
        assert enc.tokens[0] == "[BOS]" and enc.tokens[-1] == "[EOS]"
        assert enc.ids[0] == 2 and enc.ids[-1] == 3

    def test_unseen_script_costs_many_tokens(self, bpe):
        text = "الذكاء الاصطناعي"
        n_chars = len(text.replace(" ", ""))
        assert len(bpe.encode(text).tokens) >= 1.5 * n_chars    # ~2 byte-tokens per Arabic letter

    def test_tokens_are_byte_level_strings(self, bpe):
        # 'é' is 2 UTF-8 bytes (0xC3 0xA9), displayed as the two characters 'Ã©'
        assert "Ã©" in "".join(bpe.encode("é").tokens)

    def test_leading_space_is_part_of_the_token(self, bpe):
        assert bpe.encode("a b").tokens[1].startswith("Ġ")

    def test_offsets_point_into_the_original_text(self, bpe):
        text = "Hello world"
        enc = bpe.encode(text)
        assert enc.offsets[-1][1] == len(text)
        assert [o[0] for o in enc.offsets] == sorted(o[0] for o in enc.offsets)
