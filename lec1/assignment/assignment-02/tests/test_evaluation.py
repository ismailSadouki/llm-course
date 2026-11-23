import pandas as pd
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing

from src.evaluation import (
    RESULT_COLUMNS,
    TokenizerAdapter,
    evaluate_sentences,
    parity,
    summarize,
    time_encoding,
)


def make_word_level(with_special_template=False):
    """a=1 b=2, everything else is [UNK]=0 -- tiny and fully predictable."""
    vocab = {"[UNK]": 0, "a": 1, "b": 2, "[CLS]": 3, "[SEP]": 4}
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = WhitespaceSplit()
    if with_special_template:
        tok.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 3), ("[SEP]", 4)]
        )
    return tok


@pytest.fixture
def tiny_df():
    return pd.DataFrame({
        "language": ["en", "en", "fr", "ar"],
        "text": ["a b", "a c d", "b", "با"],       # 'با' = 2 chars, 4 bytes, 1 word
    })


@pytest.fixture
def tiny_results(tiny_df):
    return evaluate_sentences(TokenizerAdapter("wl", make_word_level()), tiny_df)


# ---------------------------------------------------------------- adapter
class TestAdapter:
    def test_raw_tokenizer(self):
        ad = TokenizerAdapter("raw", make_word_level())
        assert ad.tokenize("a b c") == ["a", "b", "[UNK]"]
        assert ad.encode("a b c") == [1, 2, 0]
        assert ad.unk_token == "[UNK]" and ad.unk_id == 0

    def test_vocab_size_counts_every_entry(self):
        assert TokenizerAdapter("raw", make_word_level()).vocab_size == 5

    def test_special_tokens_are_never_added(self):
        raw = make_word_level(with_special_template=True)
        assert raw.encode("a b").ids == [3, 1, 2, 4]            # the tokenizer itself adds them...
        assert TokenizerAdapter("raw", raw).encode("a b") == [1, 2]   # ...the adapter must not
        assert TokenizerAdapter("raw", raw).tokenize("a b") == ["a", "b"]

    def test_transformers_tokenizer_behaves_like_the_raw_one(self, tmp_path):
        from transformers import PreTrainedTokenizerFast

        raw = make_word_level(with_special_template=True)
        raw.save(str(tmp_path / "t.json"))
        fast = PreTrainedTokenizerFast(tokenizer_file=str(tmp_path / "t.json"), unk_token="[UNK]")
        a_raw, a_fast = TokenizerAdapter("raw", raw), TokenizerAdapter("fast", fast)
        for text in ["a b", "a c d", "b b b"]:
            assert a_raw.tokenize(text) == a_fast.tokenize(text)
            assert a_raw.encode(text) == a_fast.encode(text)
        assert a_raw.vocab_size == a_fast.vocab_size == 5
        assert a_fast.unk_id == 0

    def test_explicit_unk_token_overrides(self):
        assert TokenizerAdapter("raw", make_word_level(), unk_token="a").unk_id == 1


# ------------------------------------------------- sentence-level results
class TestEvaluateSentences:
    def test_columns_and_rows(self, tiny_results, tiny_df):
        assert list(tiny_results.columns) == RESULT_COLUMNS
        assert len(tiny_results) == len(tiny_df)
        assert (tiny_results["tokenizer"] == "wl").all()

    def test_counts(self, tiny_results):
        r = tiny_results
        assert r["n_tokens"].tolist() == [2, 3, 1, 1]
        assert r["n_unk"].tolist() == [0, 2, 0, 1]
        assert r["n_words"].tolist() == [2, 3, 1, 1]
        assert r["n_chars"].tolist() == [3, 5, 1, 2]
        assert r["n_bytes"].tolist() == [3, 5, 1, 4]

    def test_language_and_sentence_are_kept(self, tiny_results):
        assert tiny_results["language"].tolist() == ["en", "en", "fr", "ar"]
        assert tiny_results["sentence"].tolist()[0] == "a b"


# ---------------------------------------------------------------- summary
class TestSummarize:
    def test_overall_metrics(self, tiny_results):
        s = summarize(tiny_results).iloc[0]
        assert s["tokenizer"] == "wl"
        assert s["avg_tokens"] == pytest.approx(7 / 4)
        assert s["max_tokens"] == 3
        assert s["chars_per_token"] == pytest.approx(11 / 7)      # total chars / total tokens
        assert s["bytes_per_token"] == pytest.approx(13 / 7)
        assert s["tokens_per_word"] == pytest.approx(7 / 7)
        assert s["unk_rate"] == pytest.approx(3 / 7)
        assert s["n_sentences"] == 4

    def test_chars_per_token_is_a_ratio_of_totals_not_a_mean_of_ratios(self, tiny_results):
        s = summarize(tiny_results).iloc[0]
        mean_of_ratios = (tiny_results["n_chars"] / tiny_results["n_tokens"]).mean()
        assert s["chars_per_token"] != pytest.approx(mean_of_ratios)

    def test_group_by_language(self, tiny_results):
        s = summarize(tiny_results, group_cols=["tokenizer", "language"]).set_index("language")
        assert set(s.index) == {"en", "fr", "ar"}
        assert s.loc["en", "avg_tokens"] == pytest.approx(2.5)
        assert s.loc["en", "max_tokens"] == 3
        assert s.loc["en", "unk_rate"] == pytest.approx(2 / 5)
        assert s.loc["fr", "n_sentences"] == 1
        assert s.loc["ar", "bytes_per_token"] == pytest.approx(4.0)

    def test_vocab_size_column_is_optional(self, tiny_results):
        assert "vocab_size" not in summarize(tiny_results).columns
        s = summarize(tiny_results, vocab_sizes={"wl": 5})
        assert s.iloc[0]["vocab_size"] == 5

    def test_several_tokenizers(self, tiny_df):
        res = pd.concat([
            evaluate_sentences(TokenizerAdapter("one", make_word_level()), tiny_df),
            evaluate_sentences(TokenizerAdapter("two", make_word_level()), tiny_df),
        ])
        s = summarize(res)
        assert list(s["tokenizer"]) == ["one", "two"]


# ----------------------------------------------------- provided helpers
class TestProvidedHelpers:
    def test_time_encoding_returns_positive_seconds(self):
        ad = TokenizerAdapter("raw", make_word_level())
        t = time_encoding(ad, ["a b"] * 50, repeats=3)
        assert isinstance(t, float) and t > 0

    def test_parity(self, tiny_results):
        s = summarize(tiny_results, group_cols=["tokenizer", "language"])
        p = parity(s, reference="en", metric="avg_tokens")
        assert p.loc["wl", "en"] == pytest.approx(1.0)
        assert p.loc["wl", "fr"] == pytest.approx(1 / 2.5)


# ------------------------------------------- the real thing, end to end
class TestWithTrainedTokenizers:
    def test_full_benchmark_runs_and_shows_the_arabic_gap(self, bpe, wordpiece, eval_df):
        results = pd.concat([
            evaluate_sentences(TokenizerAdapter("bpe", bpe), eval_df),
            evaluate_sentences(TokenizerAdapter("wordpiece", wordpiece), eval_df),
        ])
        assert len(results) == 600
        s = summarize(results, ["tokenizer", "language"]).set_index(["tokenizer", "language"])
        # byte-level BPE never fails but pays a lot for a script it has not seen
        assert s.loc[("bpe", "ar_msa"), "unk_rate"] == 0
        assert s.loc[("bpe", "ar_msa"), "tokens_per_word"] > 3 * s.loc[("bpe", "en"), "tokens_per_word"]
        # WordPiece looks "efficient" on Arabic ... only because everything is [UNK]
        assert s.loc[("wordpiece", "ar_msa"), "unk_rate"] > 0.9
        assert s.loc[("wordpiece", "ar_msa"), "avg_tokens"] < s.loc[("wordpiece", "en"), "avg_tokens"]
        # and both are fine on the languages they were trained on
        assert s.loc[("wordpiece", "en"), "unk_rate"] < 0.02
        assert s.loc[("bpe", "fr"), "chars_per_token"] > 2
