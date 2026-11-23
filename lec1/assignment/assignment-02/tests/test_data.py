import re

import pandas as pd


def test_eval_set_shape_and_languages(eval_df):
    assert list(eval_df.columns) == ["id", "language", "text"]
    assert len(eval_df) == 300
    counts = eval_df["language"].value_counts().to_dict()
    assert counts == {"en": 100, "fr": 100, "ar_msa": 50, "ar_darija": 50}


def test_eval_sentences_are_clean_and_unique(eval_df):
    assert eval_df["text"].notna().all()
    assert (eval_df["text"].str.strip() == eval_df["text"]).all()
    assert eval_df["text"].is_unique


def test_arabic_rows_are_mostly_arabic_script(eval_df):
    arabic = re.compile(r"[\u0600-\u06FF]")
    for text in eval_df[eval_df.language.str.startswith("ar")]["text"]:
        letters = [c for c in text if c.isalpha()]
        assert sum(bool(arabic.match(c)) for c in letters) / len(letters) > 0.9


def test_training_corpus_is_big_enough(corpus_file):
    lines = corpus_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 5000
    assert all(l.strip() for l in lines)


def test_evaluation_sentences_are_not_in_the_training_corpus(corpus_file, eval_df):
    train = set(corpus_file.read_text(encoding="utf-8").splitlines())
    leaked = [t for t in eval_df["text"] if t in train]
    assert leaked == []
