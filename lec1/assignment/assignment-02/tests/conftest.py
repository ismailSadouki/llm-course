import pathlib

import pandas as pd
import pytest

from src.train_bpe import train_bpe
from src.train_wordpiece import train_wordpiece

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOY = ROOT / "data" / "toy" / "toy_multilingual.txt"
CORPUS = ROOT / "data" / "processed" / "train_corpus.txt"
EVAL = ROOT / "data" / "eval" / "eval_sentences.csv"

# Sentences used in the "same before / after reloading" tests (20 of them)
SENTENCES = [
    "Hello, how are you?",
    "Large Language Models are fascinating.",
    "Tokenization is essential for LLMs.",
    "The model is learning from data.",
    "I am studying artificial intelligence.",
    "Café, crème brûlée et œufs à la neige !",
    "L'apprentissage automatique est passionnant.",
    "Les modèles de langage comprennent-ils le contexte ?",
    "I'm learning NLP!!!",
    "HELLO WORLD",
    "Python 3.12 is great!",
    "state-of-the-art models",
    "الذكاء الاصطناعي يتطور بسرعة كبيرة.",
    "واش راك؟ راني نتعلم نماذج اللغة الكبيرة.",
    "Hello   world",
    "email me at name@example.com",
    "日本語のテキスト 🙂",
    "The price is 1,250.50 euros.",
    "   leading and trailing spaces   ",
    "Zażółć gęślą jaźń",
]


@pytest.fixture(scope="session")
def toy_file():
    return TOY


@pytest.fixture(scope="session")
def corpus_file():
    return CORPUS


@pytest.fixture(scope="session")
def eval_df():
    return pd.read_csv(EVAL)


@pytest.fixture(scope="session")
def sentences():
    return list(SENTENCES)


@pytest.fixture(scope="session")
def bpe(corpus_file):
    """Byte-level BPE, 2000 requested, trained on the bundled en+fr corpus."""
    return train_bpe(corpus_file, vocab_size=2000, min_frequency=2)


@pytest.fixture(scope="session")
def wordpiece(corpus_file):
    """WordPiece, 2000 requested, trained on the bundled en+fr corpus."""
    return train_wordpiece(corpus_file, vocab_size=2000, min_frequency=2)
