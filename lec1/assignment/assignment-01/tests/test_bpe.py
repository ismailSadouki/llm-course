import pathlib
from collections import Counter

import pytest

from src.bpe import END_OF_WORD, BPETokenizer
from src.evaluation import avg_tokens_per_word

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOY = (ROOT / "data" / "toy" / "toy_corpus.txt").read_text(encoding="utf-8").splitlines()
CLASSIC = (ROOT / "data" / "toy" / "toy_corpus_classic.txt").read_text(encoding="utf-8").splitlines()
W = END_OF_WORD


# ------------------------------------------------------------ Step 1
class TestInitializeVocabulary:
    def test_words_become_symbol_tuples_with_end_marker(self):
        vocab = BPETokenizer().initialize_vocabulary(TOY)
        assert vocab[("l", "o", "w", W)] == 3
        assert vocab[("l", "o", "w", "e", "r", W)] == 2
        assert vocab[("n", "e", "w", "e", "r", W)] == 2
        assert vocab[("n", "e", "w", "e", "s", "t", W)] == 2
        assert len(vocab) == 4

    def test_accepts_a_single_string(self):
        vocab = BPETokenizer().initialize_vocabulary("a a b")
        assert vocab == {("a", W): 2, ("b", W): 1}

    def test_multiple_lines_are_pooled(self):
        vocab = BPETokenizer().initialize_vocabulary(["a b", "a"])
        assert vocab[("a", W)] == 2

    def test_extra_whitespace_is_ignored(self):
        vocab = BPETokenizer().initialize_vocabulary("  a   b \n a ")
        assert vocab == {("a", W): 2, ("b", W): 1}


# ------------------------------------------------------------ Step 2
class TestPairFrequencies:
    def test_weighted_by_word_frequency(self):
        vocab = {
            ("l", "o", "w", W): 3,
            ("l", "o", "w", "e", "r", W): 2,
        }
        pairs = BPETokenizer().get_pair_frequencies(vocab)
        assert pairs[("l", "o")] == 5
        assert pairs[("o", "w")] == 5
        assert pairs[("w", W)] == 3
        assert pairs[("w", "e")] == 2
        assert pairs[("e", "r")] == 2
        assert pairs[("r", W)] == 2
        assert sum(pairs.values()) == 3 * 3 + 5 * 2

    def test_repeated_pair_inside_one_word_counts_each_time(self):
        pairs = BPETokenizer().get_pair_frequencies({("a", "b", "a", "b", W): 1})
        assert pairs[("a", "b")] == 2

    def test_toy_corpus_most_frequent_pair(self):
        tok = BPETokenizer()
        pairs = tok.get_pair_frequencies(tok.initialize_vocabulary(TOY))
        assert pairs[("w", "e")] == 6
        assert pairs.most_common(1)[0][0] == ("w", "e")

    def test_no_pairs_for_single_symbol_words(self):
        assert not BPETokenizer().get_pair_frequencies({("x",): 5})


# ------------------------------------------------------------ Step 4
class TestMergePair:
    def test_basic_merge(self):
        out = BPETokenizer().merge_pair(("l", "o"), {("l", "o", "w", W): 3})
        assert out == {("lo", "w", W): 3}

    def test_every_occurrence_is_merged(self):
        out = BPETokenizer().merge_pair(("a", "b"), {("a", "b", "c", "a", "b", W): 1})
        assert out == {("ab", "c", "ab", W): 1}

    def test_words_without_the_pair_are_untouched(self):
        vocab = {("x", "y", W): 4}
        assert BPETokenizer().merge_pair(("a", "b"), vocab) == vocab

    def test_input_is_not_mutated(self):
        vocab = {("a", "b", W): 1}
        BPETokenizer().merge_pair(("a", "b"), vocab)
        assert vocab == {("a", "b", W): 1}

    def test_overlapping_pairs_are_merged_left_to_right(self):
        out = BPETokenizer().merge_pair(("a", "a"), {("a", "a", "a", W): 1})
        assert out == {("aa", "a", W): 1}

    def test_total_frequency_is_preserved(self):
        vocab = {("a", "b", W): 2, ("a", "b", "c", W): 3, ("c", W): 1}
        out = BPETokenizer().merge_pair(("a", "b"), vocab)
        assert sum(out.values()) == sum(vocab.values())

    def test_end_marker_can_be_part_of_a_merge(self):
        out = BPETokenizer().merge_pair(("w", W), {("l", "o", "w", W): 3})
        assert out == {("l", "o", "w" + W): 3}


# ------------------------------------------------------- Steps 3 + 5
class TestTrain:
    def test_first_merges_on_toy_corpus(self):
        tok = BPETokenizer().train(TOY, num_merges=3)
        # (w,e)=6, then (l,o)=5, then a 4-way tie broken alphabetically -> (e,we)
        assert tok.merges == [("w", "e"), ("l", "o"), ("e", "we")]

    def test_merge_frequencies_are_recorded(self):
        tok = BPETokenizer().train(TOY, num_merges=3)
        assert tok.merge_frequencies == [6, 5, 4]

    def test_merge_frequencies_never_increase(self):
        tok = BPETokenizer().train(TOY, num_merges=50)
        f = tok.merge_frequencies
        assert all(a >= b for a, b in zip(f, f[1:]))

    def test_returns_self_for_chaining(self):
        tok = BPETokenizer()
        assert tok.train(TOY, num_merges=1) is tok

    def test_requested_number_of_merges(self):
        assert len(BPETokenizer().train(TOY, num_merges=10).merges) == 10

    def test_stops_when_no_pairs_remain(self):
        tok = BPETokenizer().train(TOY, num_merges=50)
        assert len(tok.merges) < 50
        # after the last merge every training word is a single token
        assert all(len(symbols) == 1 for symbols in tok.word_freqs)

    def test_zero_merges(self):
        tok = BPETokenizer().train(TOY, num_merges=0)
        assert tok.merges == []
        assert tok.encode("low") == ["l", "o", "w", W]

    def test_training_is_deterministic(self):
        a = BPETokenizer().train(TOY, num_merges=20).merges
        b = BPETokenizer().train(TOY, num_merges=20).merges
        assert a == b

    def test_retraining_starts_from_scratch(self):
        tok = BPETokenizer()
        tok.train(TOY, num_merges=5)
        tok.train(["a a a b"], num_merges=1)
        assert tok.merges == [("a", W)]

    def test_classic_example_first_merges(self):
        tok = BPETokenizer().train(CLASSIC, num_merges=3)
        assert tok.merges == [("e", "s"), ("es", "t"), ("est", W)]
        assert tok.merge_frequencies == [9, 9, 9]

    def test_vocabulary_size(self):
        tok = BPETokenizer().train(TOY, num_merges=10)
        base = {"l", "o", "w", "e", "r", "n", "s", "t", W}
        assert tok.vocab_size == 1 + len(base) + 10      # [UNK] + base + merges

    def test_vocabulary_layout(self):
        tok = BPETokenizer().train(TOY, num_merges=2)
        assert tok.token_to_id["[UNK]"] == 0
        assert tok.id_to_token[tok.token_to_id["we"]] == "we"
        assert tok.token_to_id["lo"] == tok.vocab_size - 1   # last merge = last id
        assert tok.token_to_id["we"] == tok.vocab_size - 2

    def test_merges_and_ranks_agree(self):
        tok = BPETokenizer().train(TOY, num_merges=8)
        for rank, pair in enumerate(tok.merges):
            assert tok.merge_ranks[pair] == rank


# ------------------------------------------------------------ Encoding
class TestEncode:
    @pytest.fixture
    def bpe30(self):
        return BPETokenizer().train(TOY, num_merges=30)

    @pytest.fixture
    def bpe10(self):
        return BPETokenizer().train(TOY, num_merges=10)

    def test_training_words_after_full_training(self, bpe30):
        for w in ["low", "lower", "newer", "newest"]:
            assert bpe30.encode_word(w) == [w + W]

    def test_encode_word_matches_training_segmentation(self, bpe10):
        # what encode_word produces for training words == what training produced
        trained = {"".join(k).replace(W, ""): k for k in bpe10.word_freqs}
        for word, symbols in trained.items():
            assert tuple(bpe10.encode_word(word)) == symbols

    def test_unseen_word_is_built_from_known_pieces(self, bpe30):
        assert bpe30.encode_word("lowest") == ["lowe", "s", "t", W]

    def test_unseen_word_differs_with_fewer_merges(self, bpe10):
        # with only 10 merges 'newest' is not a single token yet
        assert len(bpe10.encode_word("newest")) > 1

    def test_unseen_characters_survive_as_single_symbols(self, bpe30):
        assert bpe30.encode_word("xyz") == ["x", "y", "z", W]

    def test_merge_order_matters(self):
        a = BPETokenizer()
        a.merges = [("b", "c"), ("a", "b")]
        a.merge_ranks = {p: i for i, p in enumerate(a.merges)}
        assert a.encode_word("abc") == ["a", "bc", W]

        b = BPETokenizer()
        b.merges = [("a", "b"), ("b", "c")]
        b.merge_ranks = {p: i for i, p in enumerate(b.merges)}
        assert b.encode_word("abc") == ["ab", "c", W]

    def test_encode_sentence(self, bpe30):
        assert bpe30.encode("low lower") == ["low" + W, "lower" + W]

    def test_encode_empty(self, bpe30):
        assert bpe30.encode("") == []
        assert bpe30.encode("   ") == []

    def test_whitespace_is_normalized(self, bpe30):
        assert bpe30.encode("low   lower\n") == bpe30.encode("low lower")

    def test_single_character_word(self, bpe30):
        assert bpe30.encode_word("l") == ["l", W]


class TestDecode:
    @pytest.fixture(params=[0, 5, 10, 30])
    def tok(self, request):
        return BPETokenizer().train(TOY, num_merges=request.param)

    @pytest.mark.parametrize(
        "text",
        ["low", "low lower newer newest", "lowest slower", "new low new", "xyz low"],
    )
    def test_roundtrip(self, tok, text):
        assert tok.decode(tok.encode(text)) == text

    def test_decode_empty(self, tok):
        assert tok.decode([]) == ""

    def test_decode_strips_trailing_space(self):
        tok = BPETokenizer().train(TOY, num_merges=30)
        assert tok.decode(["low" + W]) == "low"


class TestIds:
    def test_tokens_to_ids_and_back(self):
        tok = BPETokenizer().train(TOY, num_merges=30)
        tokens = tok.encode("low newer")
        ids = tok.tokens_to_ids(tokens)
        assert tok.ids_to_tokens(ids) == tokens

    def test_unknown_symbols_map_to_unk_id(self):
        tok = BPETokenizer().train(TOY, num_merges=30)
        assert tok.tokens_to_ids(tok.encode("z")) == [0, tok.token_to_id[W]]


# ---------------------------------------------- Experiment-level properties
class TestExperiments:
    def test_more_merges_never_increase_tokens_per_word(self):
        texts = ["low lower newer newest lowest slower"]
        values = [
            avg_tokens_per_word(BPETokenizer().train(TOY, num_merges=n), texts)
            for n in (0, 5, 10, 13)
        ]
        assert all(a >= b for a, b in zip(values, values[1:]))
        assert values[0] > values[-1]

    def test_merges_saturate_on_tiny_corpus(self):
        n30 = BPETokenizer().train(TOY, num_merges=30)
        n50 = BPETokenizer().train(TOY, num_merges=50)
        assert n30.merges == n50.merges


@pytest.mark.skipif(
    not (ROOT / "data" / "real" / "tinyshakespeare.txt").exists(),
    reason="real dataset not downloaded (run scripts/download_data.py)",
)
class TestOnRealData:
    @pytest.fixture
    def tok_and_lines(self):
        path = ROOT / "data" / "real" / "tinyshakespeare.txt"
        lines = path.read_text(encoding="utf-8").splitlines()[:1500]
        return BPETokenizer().train(lines, num_merges=60), lines

    def test_trains_requested_merges(self, tok_and_lines):
        tok, _ = tok_and_lines
        assert len(tok.merges) == 60

    def test_roundtrip_on_real_lines(self, tok_and_lines):
        tok, lines = tok_and_lines
        for line in lines[:200]:
            assert tok.decode(tok.encode(line)) == " ".join(line.split())

    def test_common_word_becomes_one_token(self, tok_and_lines):
        tok, _ = tok_and_lines
        assert len(tok.encode_word("the")) <= 2

    def test_all_tokens_are_in_vocab(self, tok_and_lines):
        tok, lines = tok_and_lines
        for line in lines[:100]:
            for t in tok.encode(line):
                assert t in tok.token_to_id
