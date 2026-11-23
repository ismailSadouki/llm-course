import pytest

from src.char_tokenizer import CharacterTokenizer
from src.word_tokenizer import WordTokenizer, detokenize, word_tokenize


# ===================================================== CharacterTokenizer
class TestCharacterTokenizer:
    @pytest.fixture
    def tok(self):
        t = CharacterTokenizer()
        t.train(["abc", "cab"])
        return t

    def test_special_token_ids_are_fixed(self, tok):
        assert [tok.token_to_id[s] for s in ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]] == [0, 1, 2, 3]

    def test_vocab_is_specials_plus_distinct_chars(self, tok):
        assert tok.vocab_size == 4 + 3

    def test_chars_are_sorted_after_specials(self, tok):
        assert [tok.id_to_token[i] for i in range(4, 7)] == ["a", "b", "c"]

    def test_encode_known(self, tok):
        assert tok.encode("abc") == [4, 5, 6]

    def test_one_id_per_character(self, tok):
        assert len(tok.encode("abccba")) == 6

    def test_unknown_characters_map_to_unk(self, tok):
        assert tok.encode("axb") == [4, 1, 5]

    def test_add_special_tokens(self, tok):
        assert tok.encode("ab", add_special_tokens=True) == [2, 4, 5, 3]

    def test_decode_roundtrip(self, tok):
        assert tok.decode(tok.encode("abcabc")) == "abcabc"

    def test_decode_skips_specials_by_default(self, tok):
        ids = tok.encode("ab", add_special_tokens=True) + [0, 0]
        assert tok.decode(ids) == "ab"

    def test_decode_can_keep_specials(self, tok):
        ids = tok.encode("ab", add_special_tokens=True)
        assert tok.decode(ids, skip_special_tokens=False) == "[BOS]ab[EOS]"

    def test_unknown_char_is_lost_on_decode(self, tok):
        assert tok.decode(tok.encode("axb")) == "ab"

    def test_invalid_id_raises(self, tok):
        with pytest.raises(ValueError):
            tok.decode([999])

    def test_empty_text(self, tok):
        assert tok.encode("") == []
        assert tok.decode([]) == ""

    def test_case_sensitive(self):
        t = CharacterTokenizer()
        t.train(["aA"])
        assert t.vocab_size == 4 + 2

    def test_space_is_a_character(self):
        t = CharacterTokenizer()
        t.train(["a b"])
        assert t.decode(t.encode("a b")) == "a b"

    def test_unicode_roundtrip(self):
        text = "واش راك؟ café ça va"
        t = CharacterTokenizer()
        t.train([text])
        assert t.decode(t.encode(text)) == text

    def test_retraining_resets_vocabulary(self):
        t = CharacterTokenizer()
        t.train(["abc"])
        t.train(["xy"])
        assert t.vocab_size == 4 + 2
        assert t.encode("a") == [1]


# ========================================================== word_tokenize
class TestWordTokenizeConvention:
    def test_punctuation_is_separate(self):
        assert word_tokenize("I love language models.") == ["I", "love", "language", "models", "."]

    def test_apostrophes_stay_inside_words(self):
        assert word_tokenize("Let's eat, grandma!") == ["Let's", "eat", ",", "grandma", "!"]

    def test_french_elision(self):
        assert word_tokenize("L'apprentissage automatique") == ["L'apprentissage", "automatique"]

    def test_case_is_preserved(self):
        assert word_tokenize("Apple apple") == ["Apple", "apple"]

    def test_detokenize_spacing(self):
        assert detokenize(["Hello", ",", "world", "!"]) == "Hello, world!"
        assert detokenize(["(", "yes", ")"]) == "(yes)"


# ============================================================ WordTokenizer
class TestWordTokenizer:
    CORPUS = ["the cat sat on the mat .", "the dog sat ."]

    @pytest.fixture
    def tok(self):
        t = WordTokenizer()
        t.train(self.CORPUS)
        return t

    def test_special_token_ids_are_fixed(self, tok):
        assert [tok.token_to_id[s] for s in ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]] == [0, 1, 2, 3]

    def test_vocab_contents(self, tok):
        # the cat sat on mat . dog  -> 7 distinct tokens
        assert tok.vocab_size == 4 + 7

    def test_most_frequent_tokens_get_smallest_ids(self, tok):
        # "the" (3x) < "sat" (2x) and "." (2x) < the rest (1x), ties alphabetical
        order = [tok.id_to_token[i] for i in range(4, 8)]
        assert order == ["the", ".", "sat", "cat"]

    def test_encode_decode_roundtrip(self, tok):
        assert tok.decode(tok.encode("the dog sat on the mat.")) == "the dog sat on the mat."

    def test_unknown_word_maps_to_unk(self, tok):
        ids = tok.encode("the bird sat")
        assert ids[1] == tok.token_to_id["[UNK]"]

    def test_unknown_word_is_lost_in_decoding(self, tok):
        assert tok.decode(tok.encode("the bird sat")) == "the sat"

    def test_add_special_tokens(self, tok):
        ids = tok.encode("the cat", add_special_tokens=True)
        assert ids[0] == 2 and ids[-1] == 3 and len(ids) == 4

    def test_decode_keeps_specials_if_asked(self, tok):
        ids = tok.encode("the cat", add_special_tokens=True)
        assert tok.decode(ids, skip_special_tokens=False).startswith("[BOS]")

    def test_min_freq(self):
        t = WordTokenizer(min_freq=2)
        t.train(self.CORPUS)
        kept = {tok for tok in t.token_to_id if tok not in t.SPECIAL_TOKENS}
        assert kept == {"the", "sat", "."}

    def test_max_vocab_size_counts_only_regular_tokens(self):
        t = WordTokenizer(max_vocab_size=3)
        t.train(self.CORPUS)
        assert t.vocab_size == 4 + 3
        assert "the" in t.token_to_id

    def test_oov_rate_counts_occurrences(self, tok):
        # tokens: the, bird, sat, ., -> 1 unknown out of 4
        assert tok.oov_rate(["the bird sat ."]) == pytest.approx(0.25)

    def test_oov_rate_on_training_data_is_zero(self, tok):
        assert tok.oov_rate(self.CORPUS) == 0.0

    def test_oov_rate_empty(self, tok):
        assert tok.oov_rate([]) == 0.0

    def test_case_sensitive_vocabulary(self):
        t = WordTokenizer()
        t.train(["Apple apple"])
        assert t.vocab_size == 4 + 2

    def test_retraining_resets_vocabulary(self):
        t = WordTokenizer()
        t.train(["a b c"])
        t.train(["x"])
        assert t.vocab_size == 4 + 1

    def test_invalid_id_raises(self, tok):
        with pytest.raises(ValueError):
            tok.decode([9999])

    def test_word_vocab_is_much_bigger_than_char_vocab(self):
        text = ["the quick brown fox jumps over the lazy dog again and again"]
        w, c = WordTokenizer(), CharacterTokenizer()
        w.train(text)
        c.train(text)
        assert w.vocab_size > 0 and c.vocab_size > 0
        # but sequences are much shorter with words
        assert len(w.encode(text[0])) < len(c.encode(text[0]))
