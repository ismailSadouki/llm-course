"""Conservative text preprocessing for LLM data pipelines   (Part A)

Policy you must implement (document any change in your report):

    1. Strip HTML            -> keep the visible text only
    2. Unicode normalization -> NFC  (NOT NFKC -- explain why in your notebook)
    3. Whitespace            -> collapse runs of whitespace into ONE space
    4. Never touched         -> punctuation, capitalization, stop words, word order

Run the tests for this part with:   pytest tests/test_preprocessing.py
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional

UNICODE_FORM = "NFC"


# --------------------------------------------------------------------------
# TODO 1 -- Unicode
# --------------------------------------------------------------------------
def normalize_unicode(text: str) -> str:
    """Return `text` in NFC form.   (= notebook 01, exercise 1.1)

    Hint: unicodedata.normalize(form, text). Try it on "e\\u0301" vs "\\u00e9".
    """
    raise NotImplementedError


# --------------------------------------------------------------------------
# TODO 2 -- HTML
# --------------------------------------------------------------------------
def remove_html(text: str) -> str:
    """Remove HTML tags and return the visible text.   (= notebook 01, exercise 1.3: complete the
    TextExtractor class there, then move it into this file)

    Requirements (all covered by the tests):
      * tags and their attributes disappear
      * HTML entities are decoded:  &amp; -> &    &lt; -> <    &nbsp; -> space
      * <script> and <style> CONTENT is dropped (it is not visible text)
      * block-level tags (<p>, <div>, <br>, <li> ...) separate text with a space
      * inline tags (<b>, <i>, <a> ...) must NOT add a space:
            "This is <b>very</b>."  ->  "This is very."
      * "5 &lt; 10" is TEXT, not a tag -- it must survive as "5 < 10"

    Hints:
      * Quick-and-dirty regexes (re.sub(r"<[^>]+>", " ", text)) will fail
        several tests. Think about WHY.
      * Python's standard library has html.parser.HTMLParser. Subclass it and
        override handle_starttag / handle_endtag / handle_data.
        Create it with HTMLParser(convert_charrefs=True) so entities are
        decoded for you.
      * Order matters: decode entities AFTER removing tags, never before.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------
# TODO 3 -- Whitespace
# --------------------------------------------------------------------------
def normalize_whitespace(text: str) -> str:
    """Strip both ends and replace every run of whitespace by one space.   (= notebook 01, exercise 1.2)

    Hint: in Python, the regex \\s already matches tabs, newlines AND the
    non-breaking space \\xa0 for str patterns.
    """
    raise NotImplementedError


# --------------------------------------------------------------------------
# TODO 4 -- One document
# --------------------------------------------------------------------------
def preprocess_text(text: Optional[str]) -> str:
    """Complete conservative pipeline for ONE document.

    * None or any non-string value -> "" (missing record)
    * Apply: remove_html -> normalize_unicode -> normalize_whitespace
      (write a comment explaining why you chose this order)
    * Must be idempotent: preprocess_text(preprocess_text(x)) == preprocess_text(x)
    """
    raise NotImplementedError


# --------------------------------------------------------------------------
# TODO 5 -- Corpus level
# --------------------------------------------------------------------------
def drop_empty(documents: Iterable[Optional[str]]) -> list[str]:
    """Remove None, non-string and blank (whitespace-only) documents."""
    raise NotImplementedError


def remove_duplicates(documents: list[str]) -> list[str]:
    """Remove EXACT duplicates, keep the FIRST occurrence, preserve order.   (= notebook 01, exercise 1.4)

    Hint: a `set` of already-seen documents gives O(n). Do not call
    list(set(documents)) -- sets do not preserve order.
    """
    raise NotImplementedError


def preprocess_corpus(documents: Iterable[Optional[str]]) -> tuple[list[str], dict]:
    """Run the whole pipeline on a corpus and report what happened.

    Steps: clean every document -> drop empty -> drop exact duplicates.
    (Duplicates are detected AFTER cleaning: "Hello  world" == "Hello world".)

    Return (clean_documents, stats) where stats has exactly these keys:
        n_input, n_empty_removed, n_duplicates_removed, n_output
    """
    raise NotImplementedError


# --------------------------------------------------------------------------
# PROVIDED (do not modify): the "classic NLP" pipeline used in Task A.3 so
# you can compare it with your conservative pipeline.
# --------------------------------------------------------------------------
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "am",
    "i", "me", "my", "you", "he", "she", "it", "we", "they",
    "of", "to", "in", "on", "at", "and", "or", "but", "not", "no",
    "this", "that", "do", "does", "did", "for", "with",
}


def aggressive_normalize(text: str) -> str:
    """Lowercase + remove punctuation + remove stop words."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(w for w in text.split() if w not in STOP_WORDS)
