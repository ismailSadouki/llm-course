# Data used in Assignment 02

| File | What | Size | Use |
|---|---|---|---|
| `processed/train_corpus.txt` | English + French training sentences, shuffled | 6 000 lines | train your BPE and WordPiece tokenizers |
| `processed/train_en.txt`, `train_fr.txt` | the same corpus split by language | 3 000 + 3 000 lines | language-specific experiments |
| `eval/eval_sentences.csv` | evaluation sentences: columns `id, language, text` | 300 rows | the benchmark (Part G) — **never train on it** |
| `toy/toy_multilingual.txt` | 20 short English/French sentences | 20 lines | fast experiments and unit tests |

## Training corpus (document this in your report!)

* **Source:** Europarl (Koehn, 2005), proceedings of the European Parliament, taken from the English and French parts of the `europarl_raw` package of the NLTK data repository (sessions from early 2000).
* **Processing:** `scripts/build_corpus.py` — SGML tags removed, split into sentences, 30–220 characters, at most 6 digits, deduplicated, **detokenized** (the NLTK copy is pre-tokenized: `"l ' ordre"`, `"hope , in"`), 3 000 sentences per language drawn with seed 0.
* **Language distribution:** 50 % English, 50 % French, **0 % Arabic or Darija**. This is deliberate: your tokenizers will meet a script they never saw, and you will measure the consequences.
* **Limitations:** formal political register, one domain, one period, no informal language, no code-switching.

## Evaluation set

* 100 English, 100 French (largely parallel to the English ones), 50 Modern Standard Arabic and 50 Algerian Darija sentences, written for this course. They are not extracted from the training corpus (a test checks it).
* Darija is written in Arabic script, with the usual spelling variation; some sentences contain French loanwords written in Arabic letters.
* Short everyday sentences, 30–100 characters.

## Optional: add your own Arabic / Darija training data

```bash
python scripts/build_corpus.py --extra my_arabic_sentences.txt       # one sentence per line
```
Check the **licence** of anything you add, keep the evaluation sentences out of it, and document the source and the language distribution in your report.
