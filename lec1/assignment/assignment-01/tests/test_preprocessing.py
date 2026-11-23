import json
import pathlib

import pytest

from src.preprocessing import (
    aggressive_normalize,
    drop_empty,
    normalize_unicode,
    normalize_whitespace,
    preprocess_corpus,
    preprocess_text,
    remove_duplicates,
    remove_html,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------- Unicode
class TestNormalizeUnicode:
    def test_nfd_becomes_nfc(self):
        decomposed = "e\u0301"          # 'e' + combining acute accent
        composed = "\u00e9"             # precomposed 'é'
        assert normalize_unicode(decomposed) == composed

    def test_already_nfc_is_unchanged(self):
        assert normalize_unicode("L'élève à l'école") == "L'élève à l'école"

    def test_idempotent(self):
        once = normalize_unicode("e\u0301a\u0300")
        assert normalize_unicode(once) == once

    def test_is_nfc_not_nfkc(self):
        # NFKC would turn the ligature into "fi" and destroy information.
        assert normalize_unicode("\ufb01") == "\ufb01"

    def test_arabic_is_preserved(self):
        text = "السَّلامُ عَلَيْكُم"
        assert normalize_unicode(text) == text


# -------------------------------------------------------------------- HTML
class TestRemoveHtml:
    def test_plain_text_is_unchanged(self):
        assert remove_html("Just text.") == "Just text."

    def test_tags_are_removed(self):
        out = normalize_whitespace(remove_html("<p>Hello <b>world</b></p>"))
        assert out == "Hello world"

    def test_inline_tags_do_not_add_spaces_before_punctuation(self):
        out = normalize_whitespace(remove_html("This is <b>very</b>. Really."))
        assert out == "This is very. Really."

    def test_block_tags_separate_text(self):
        out = normalize_whitespace(remove_html("<p>First</p><p>Second</p>"))
        assert out == "First Second"

    def test_br_separates_text(self):
        out = normalize_whitespace(remove_html("line one<br>line two<br/>line three"))
        assert out == "line one line two line three"

    def test_entities_are_decoded(self):
        out = normalize_whitespace(remove_html("Tom &amp; Jerry &copy; 2024"))
        assert out == "Tom & Jerry © 2024"

    def test_script_and_style_content_is_dropped(self):
        html = "<script>var x = 1;</script><style>p{color:red}</style><div>Visible.</div>"
        assert normalize_whitespace(remove_html(html)) == "Visible."

    def test_escaped_angle_brackets_are_text_not_tags(self):
        # '&lt; 10 &amp; 10 &gt;' must survive: it is TEXT, not a tag.
        out = normalize_whitespace(remove_html("5 &lt; 10 &amp; 10 &gt; 5"))
        assert out == "5 < 10 & 10 > 5"

    def test_attributes_do_not_leak(self):
        out = normalize_whitespace(remove_html('<a href="http://x.org" class="c">link</a>'))
        assert out == "link"


# -------------------------------------------------------------- Whitespace
class TestNormalizeWhitespace:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("  hello  ", "hello"),
            ("a   b", "a b"),
            ("a\tb\nc", "a b c"),
            ("a\u00a0b", "a b"),       # non-breaking space
            ("", ""),
            ("   \n ", ""),
            ("already clean", "already clean"),
        ],
    )
    def test_cases(self, raw, expected):
        assert normalize_whitespace(raw) == expected


# ----------------------------------------------------- Full single document
class TestPreprocessText:
    @pytest.mark.parametrize("bad", [None, "", "   ", "\n\t", 12345, 3.14, ["a"]])
    def test_missing_or_empty_returns_empty_string(self, bad):
        assert preprocess_text(bad) == ""

    def test_preserves_capitalization_punctuation_and_stop_words(self):
        s = "The model is NOT performing well!"
        assert preprocess_text(s) == s

    def test_preserves_word_order(self):
        s = "To be, or not to be: that is the question."
        assert preprocess_text(s) == s

    def test_cleans_html_unicode_and_whitespace_together(self):
        raw = "  <p>Le  mod\u0065\u0300le\u00a0est <b>bon</b>.</p>\n"
        assert preprocess_text(raw) == "Le mod\u00e8le est bon."

    def test_is_idempotent(self):
        raw = "<div> Hello \u00a0  <i>World</i> </div>"
        once = preprocess_text(raw)
        assert preprocess_text(once) == once

    def test_arabic_diacritics_and_tatweel_are_kept(self):
        s = "السَّلامُ مـــن هنا"
        assert preprocess_text(s) == s

    def test_different_meanings_stay_different(self):
        a = preprocess_text("Let's eat, grandma!")
        b = preprocess_text("Let's eat grandma!")
        assert a != b


# ------------------------------------------------------------- Dedup/empty
class TestDocumentFilters:
    def test_drop_empty(self):
        docs = ["a", None, "", "  ", "b", 5]
        assert drop_empty(docs) == ["a", "b"]

    def test_remove_duplicates_keeps_first_and_order(self):
        assert remove_duplicates(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]

    def test_remove_duplicates_is_case_sensitive(self):
        assert remove_duplicates(["Apple", "apple", "Apple"]) == ["Apple", "apple"]

    def test_remove_duplicates_empty_input(self):
        assert remove_duplicates([]) == []

    def test_remove_duplicates_does_not_mutate_input(self):
        docs = ["a", "a"]
        remove_duplicates(docs)
        assert docs == ["a", "a"]


class TestPreprocessCorpus:
    def test_duplicates_are_found_after_cleaning(self):
        docs = ["Hello   world", "Hello world", "<p>Hello world</p>"]
        out, stats = preprocess_corpus(docs)
        assert out == ["Hello world"]
        assert stats["n_duplicates_removed"] == 2

    def test_stats_are_consistent(self):
        docs = ["a", None, "", "a", "b"]
        out, stats = preprocess_corpus(docs)
        assert out == ["a", "b"]
        assert stats == {
            "n_input": 5,
            "n_empty_removed": 2,
            "n_duplicates_removed": 1,
            "n_output": 2,
        }

    def test_messy_dataset(self):
        path = ROOT / "data" / "raw" / "messy_documents.jsonl"
        with open(path, encoding="utf-8") as f:
            docs = [json.loads(line)["text"] for line in f]
        out, stats = preprocess_corpus(docs)
        assert stats == {
            "n_input": 24,
            "n_empty_removed": 4,
            "n_duplicates_removed": 5,
            "n_output": 15,
        }
        # the NFD and NFC versions of the French sentence collapsed into one
        assert out.count("L'élève à l'école.") == 1
        # sentences that differ only by punctuation / case are NOT merged
        assert "The model is NOT performing well!" in out
        assert "The model is not performing well." in out
        assert "Let's eat, grandma!" in out and "Let's eat grandma!" in out
        assert "Price: 5 < 10 & 10 > 5" in out


# ---------------------------------------- Why aggressive cleaning is risky
class TestAggressiveVsConservative:
    PAIRS = [
        ("The model is NOT performing well!", "The model is not performing well."),
        ("Let's eat, grandma!", "Let's eat grandma!"),
        ("Apple released a new product.", "I ate an apple yesterday."),
    ]

    def test_aggressive_merges_distinct_sentences(self):
        a, b = self.PAIRS[0]
        assert aggressive_normalize(a) == aggressive_normalize(b)
        a, b = self.PAIRS[1]
        assert aggressive_normalize(a) == aggressive_normalize(b)

    def test_conservative_keeps_them_distinct(self):
        for a, b in self.PAIRS[:2]:
            assert preprocess_text(a) != preprocess_text(b)
