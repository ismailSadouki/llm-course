import pytest

from src.train_wordpiece import (
    DEFAULT_SPECIAL_TOKENS,
    build_wordpiece_tokenizer,
    train_wordpiece,
    train_wordpiece_from_texts,
)


class TestUntrainedTokenizer:
    def test_components(self):
        tok = build_wordpiece_tokenizer()
        assert type(tok.model).__name__ == "WordPiece"
        assert type(tok.normalizer).__name__ == "BertNormalizer"
        assert type(tok.pre_tokenizer).__name__ == "BertPreTokenizer"
        assert type(tok.decoder).__name__ == "WordPiece"


class TestTraining:
    def test_special_tokens_and_ids(self, wordpiece):
        assert DEFAULT_SPECIAL_TOKENS == ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
        assert [wordpiece.token_to_id(t) for t in DEFAULT_SPECIAL_TOKENS] == [0, 1, 2, 3, 4]

    def test_vocab_size_is_reached(self, wordpiece):
        assert wordpiece.get_vocab_size() == 2000

    def test_vocab_can_be_smaller_than_requested(self, toy_file):
        assert train_wordpiece(toy_file, vocab_size=5000).get_vocab_size() < 5000

    def test_continuation_pieces_exist_in_vocabulary(self, wordpiece):
        assert sum(t.startswith("##") for t in wordpiece.get_vocab()) > 100

    def test_min_frequency_shrinks_the_vocabulary_of_a_small_corpus(self, toy_file):
        low = train_wordpiece(toy_file, vocab_size=5000, min_frequency=1).get_vocab_size()
        high = train_wordpiece(toy_file, vocab_size=5000, min_frequency=6).get_vocab_size()
        assert high < low

    def test_train_from_texts_gives_almost_the_same_vocabulary_as_from_file(self, toy_file):
        # WordPiece training in `tokenizers` is NOT perfectly reproducible from one run to the
        # next (ties are broken arbitrarily): compare overlap, not equality. This is one more
        # reason to SAVE a trained tokenizer instead of retraining it.
        lines = toy_file.read_text(encoding="utf-8").splitlines()
        a = set(train_wordpiece(toy_file, vocab_size=400).get_vocab())
        b = set(train_wordpiece_from_texts(lines, vocab_size=400).get_vocab())
        assert len(a & b) / len(a | b) > 0.8


class TestNormalization:
    def test_lowercases(self, wordpiece):
        assert wordpiece.encode("HELLO World").tokens == wordpiece.encode("hello world").tokens

    def test_strips_accents(self, wordpiece):
        assert wordpiece.encode("CAFÉ").tokens == wordpiece.encode("cafe").tokens

    def test_accents_can_be_kept(self):
        tok = train_wordpiece_from_texts(["été été été été été"] * 20, vocab_size=100,
                                         min_frequency=1, strip_accents=False)
        assert tok.encode("été").tokens != tok.encode("ete").tokens

    def test_case_can_be_kept(self):
        tok = train_wordpiece_from_texts(["The the The the The the"] * 20, vocab_size=100,
                                         min_frequency=1, lowercase=False)
        assert tok.encode("The").tokens != tok.encode("the").tokens


class TestSubwords:
    WORD = "incomprehensibilities"

    def test_rare_word_is_split_with_continuation_prefix(self, wordpiece):
        tokens = wordpiece.encode(self.WORD).tokens
        assert len(tokens) > 1
        assert not tokens[0].startswith("##")
        assert all(t.startswith("##") for t in tokens[1:])

    def test_decoder_glues_the_pieces_back(self, wordpiece):
        assert wordpiece.decode(wordpiece.encode(self.WORD).ids) == self.WORD

    def test_decode_is_lossy(self, wordpiece):
        # lower-casing and accent stripping cannot be undone
        assert wordpiece.decode(wordpiece.encode("Café").ids) == "cafe"

    def test_punctuation_is_separated_and_decoded_cleanly(self, wordpiece):
        enc = wordpiece.encode("Hello, world!")
        assert enc.tokens[-1] == "!" and "," in enc.tokens
        assert wordpiece.decode(enc.ids) == "hello, world!"


class TestUnknowns:
    def test_unseen_script_becomes_one_unk_per_word(self, wordpiece):
        assert wordpiece.encode("واش راك").ids == [0, 0]
        assert wordpiece.encode("واش راك").tokens == ["[UNK]", "[UNK]"]

    def test_special_tokens_are_atomic(self, wordpiece):
        enc = wordpiece.encode("[CLS] hello [SEP]")
        assert enc.ids[0] == 2 and enc.ids[-1] == 3
