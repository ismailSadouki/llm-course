# Assignment 02 — Text Processing and Tokenization with Hugging Face
### Large Language Models: Foundations and Engineering

| | |
|---|---|
| **Type** | Practical programming + experimental analysis |
| **Difficulty** | Intermediate |
| **Workload** | 8–12 hours |
| **Team** | Individual |
| **Prerequisite** | Assignment 01 (you built BPE yourself; now you meet the industrial tools) |
| **Submission** | Your GitHub repository (this repo, completed) + `report.pdf` |
| **Needs** | Python 3.10+, **internet access** (first download of the pretrained tokenizers), no GPU |

---

## 1. What is this assignment about?

In Assignment 01 you implemented tokenization algorithms from scratch. Here you use the professional tools — Hugging Face `tokenizers`, `transformers` and `datasets` — and learn to *look inside* them:

```
pretrained tokenizers ──► explore (vocabulary, normalization, pre-tokenization)
your own corpus ────────► train BPE + WordPiece ──► save / reload / wrap for Transformers
all of them ────────────► benchmark on English, French, Arabic and Algerian Darija
```

The questions you will answer with numbers:

* What exactly is inside a tokenizer, and which parts are *not* the vocabulary?
* What does a tokenizer do to text before and after the main algorithm (normalizer, pre-tokenizer, decoder)?
* What happens when a tokenizer meets a language or a script it was not trained on?
* Is "fewer tokens" the same as "better"? (spoiler: no)

**Scope.** Text processing and tokenization only. Transformer architectures, attention, embeddings, inference and training are out of scope.

## 2. Learning objectives

1. Load pretrained tokenizers and read their configuration, vocabulary and special tokens.
2. Convert text ↔ tokens ↔ ids and explain every difference in the round trip.
3. Explain normalization and pre-tokenization, and show their effect on the output.
4. Train byte-level BPE and WordPiece tokenizers with `tokenizers`.
5. Save, reload and wrap a tokenizer (`PreTrainedTokenizerFast`) and verify that it behaves identically.
6. Build a fair, reusable multilingual tokenizer evaluation pipeline and interpret its metrics critically.

## 3. Setup

```bash
git clone <your-repo-url> assignment-02 && cd assignment-02
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import tokenizers, transformers, datasets; print(tokenizers.__version__, transformers.__version__, datasets.__version__)"
pytest                      # red where code is missing; data tests are already green
jupyter notebook
```

* The first call to `AutoTokenizer.from_pretrained(...)` downloads files from the Hugging Face Hub (a few MB each) and caches them. Tests that need the pretrained tokenizers are **skipped** automatically if the download fails.
* The `transformers` API changes between major versions. If something in the handout does not match what you see, check your version and the documentation, and mention it in the report.

## 4. How to work

The assignment has **two kinds of programming work**, and they feed each other:

| | What | Where | Feedback |
|---|---|---|---|
| 🧩 **Code-completion exercises** (26) | Fill the `____` blanks with the Hugging Face code you will use in every LLM project: loading tokenizers, `tokenize`/`encode`/`decode`, normalizers, pre-tokenizers, building a BPE/WordPiece pipeline, trainers, `train`, saving, `PreTrainedTokenizerFast`, … Every exercise has **hints** (names of the functions to use + documentation links) | the notebooks | a ✅ *check* cell right after each exercise |
| 🛠 **Functions in `src/`** | Package what you learned into reusable, tested functions | `src/*.py` | `pytest` |

Typical loop: **do the 🧩 exercises of a notebook → move the working code into the matching `src/` function (docstrings tell you which exercise it comes from) → `pytest` → finish the notebook (experiments + written answers).**

**How a 🧩 exercise works.** You see something like this:

```python
tokenizer = Tokenizer(____)
tokenizer.pre_tokenizer = ____
tokenizer.decoder = ____
```
Read the hints above the cell, replace each `____`, run the cell, then run the ✅ check cell. If the check fails, its error message tells you what is wrong. Do not delete the check cells. Do not look for solutions online: the hints plus the documentation are enough, and you will be asked to explain your code.

| Step | 🧩 Exercises | Files to complete | Tests |
|---|---|---|---|
| Explore pretrained tokenizers | notebooks 01–05 (13 exercises) | — | `tests/test_pretrained.py` (needs internet) |
| Train BPE | notebook 06, ex. 6.1–6.3 | `src/train_bpe.py` | `tests/test_bpe.py` |
| Train WordPiece | notebook 06, ex. 6.4–6.6 | `src/train_wordpiece.py` | `tests/test_wordpiece.py` |
| Save, reload, wrap | notebook 06, ex. 6.8–6.9 | `src/tokenizer_io.py` | `tests/test_tokenizer_io.py` |
| Benchmark | notebook 07, ex. 7.1–7.4 | `src/evaluation.py` | `tests/test_evaluation.py` |
| Data sanity | — | — | `tests/test_data.py` |

Notebook 06 is in two phases: first the 🧩 exercises (they do not need `src/`), then — after you have packaged the code and the tests pass — the experiments that use your functions. Notebook 07 uses your `src/` code from the start. Do not edit the provided tests (you may add your own in new files — the assignment asks you to write tests for your tokenizers, and the provided ones are a minimum).

### Repository map

| Path | Status | Content |
|---|---|---|
| `src/train_bpe.py`, `train_wordpiece.py` | **TODO** | build + train functions, CLI provided |
| `src/tokenizer_io.py` | **TODO** | save / load / `PreTrainedTokenizerFast` wrapper / consistency check |
| `src/evaluation.py` | **TODO** (part) | adapter, per-sentence results, summary; timing and parity provided |
| `src/pretrained.py` | provided | loads GPT-2, BERT, FLAN-T5 (skips what it cannot load) |
| `data/` | provided | training corpus, 300 evaluation sentences, toy file — read `data/README.md` |
| `notebooks/01…07` | **TODO** | 26 🧩 code-completion exercises with hints and check cells + guided experiments + your written answers |
| `models/bpe`, `models/wordpiece` | you | your saved tokenizers (commit them) |
| `results/` | you | CSV tables and figures produced by notebook 07 |
| `docs/WALKTHROUGH.md` | provided | explanatory guide to every part (read it first) |
| `report/REPORT_TEMPLATE.md` | provided | structure of the report |

## 5. Grading

| Part | Topic | Pts | Assessed by |
|---|---|---:|---|
| A | Hugging Face ecosystem | 5 | notebook 01 written answers |
| B | Exploring pretrained tokenizers | 10 | notebook 01: exercises 1.1–1.3 (5) + answers (5) |
| C | Encoding, decoding, vocabulary | 10 | notebooks 02, 03: exercises (5) + answers (5) |
| D | Normalization and pre-tokenization | 10 | notebooks 04, 05: exercises (5) + answers (5) |
| E | Training BPE and WordPiece | 20 | tests (10) + notebook 06 exercises 6.1–6.7 (4) + experiments and answers (6) |
| F | Saving, loading, automated tests | 10 | tests (4) + exercises 6.8–6.9 (2) + your own tests (4) |
| G | Multilingual benchmark | 25 | tests (7) + exercises 7.1–7.4 (3) + notebook 07 results and analysis (15) |
| H | Report, final challenge, code quality | 10 | report + repo |
| | **Total** | **100** | |

**Rules.** Notebooks must re-run top to bottom. Every number in your report comes from your code. Cite your sources (documentation, papers, AI assistants if you used them). You must be able to explain your code in a short oral check. Hidden tests may be used; they test only what the docstrings state.

---

## 6. The tasks

### Part A — The Hugging Face ecosystem (5 pts) · notebook 01
Read the documentation of `tokenizers`, `transformers`, `datasets`, `huggingface_hub` and answer the seven questions in the notebook (purpose of each library, `tokenizers` vs `transformers`, `AutoTokenizer`, pretrained tokenizer, tokenizer configuration, the Hub, fast vs slow tokenizers).

### Part B — Pretrained tokenizers (10 pts) · notebook 01
🧩 Exercises 1.1–1.3: load with `AutoTokenizer.from_pretrained`, interrogate (`vocab_size`, `len`, `special_tokens_map`, `is_fast`), look inside (`backend_tokenizer`). Load **GPT-2** (byte-level BPE), **BERT-base-uncased** (WordPiece) and **FLAN-T5-small** (SentencePiece Unigram). For each: class, vocabulary size, special tokens, fast or slow, configuration files, the components of its backend tokenizer, and why `vocab_size` and `len(tokenizer)` can differ. Answer the five questions.

### Part C — Encoding, decoding, vocabulary (10 pts) · notebooks 02, 03
🧩 Exercises 2.1–2.3 (`tokenize`, `convert_tokens_to_ids`, `convert_ids_to_tokens`, `decode`, `encode(..., add_special_tokens=False)`, batch + `padding=True`) and 3.1–3.3 (`get_vocab`, inverting a vocabulary, lookups, finding multi-piece words). Tokenize five sentences with the three tokenizers; show tokens, ids, ids → tokens, decoded text; explain every difference between original and decoded text. Then extract each vocabulary, look at 50 tokens, convert word ↔ id, find multi-piece words, and answer: are vocabularies only complete words? are ids meaningful rankings? what do `Ġ`, `▁`, `##` mean?

### Part D — Normalization and pre-tokenization (10 pts) · notebooks 04, 05
🧩 Exercises 4.1–4.2 (read a tokenizer's normalizer; compose `NFD`, `StripAccents`, `Lowercase` in a `Sequence`; `BertNormalizer`) and 5.1–5.2 (create `Whitespace`, `WhitespaceSplit`, `BertPreTokenizer`, `ByteLevel`; combine pre-tokenizers). Compare how the tokenizers treat capitalization, accents and punctuation on eight given inputs; inspect their normalizers; build normalizers yourself with `tokenizers.normalizers`. Then run `Whitespace`, `WhitespaceSplit`, `BertPreTokenizer` and `ByteLevel` on five inputs and explain how punctuation and spaces are represented and why pre-tokenization changes the final tokens.

### Part E — Training custom tokenizers (20 pts) · `src/train_bpe.py`, `src/train_wordpiece.py`, notebook 06
🧩 **First write the code in notebook 06** (exercises 6.1–6.7: assemble the pipeline, configure the trainer, `train`, inspect `tokens/ids/offsets`, WordPiece pipeline, `train_from_iterator`, `TemplateProcessing` post-processor) — **then package it** by completing the functions that build and train:

* a **byte-level BPE** (BPE model, ByteLevel pre-tokenizer and decoder, `BpeTrainer` with the 256-byte initial alphabet), and
* a **WordPiece** tokenizer (`BertNormalizer`, `BertPreTokenizer`, `WordPieceTrainer`, `##` decoder),

with vocabulary size **2 000**, on the bundled corpus (`data/processed/train_corpus.txt`; document its source and language distribution). In notebook 06: display the learned vocabulary, tokenize and decode the same 20 sentences with both tokenizers, study *requested vs. final vocabulary size* and `min_frequency`, and run the reproducibility experiment (train the same tokenizer several times — are the vocabularies identical?).

### Part F — Saving, reloading, testing (10 pts) · `src/tokenizer_io.py`, notebook 06, your own tests
🧩 Exercises 6.8–6.9 (`save`, `Tokenizer.from_file`, `PreTrainedTokenizerFast`, `save_pretrained`, `AutoTokenizer.from_pretrained`), then package them in `src/tokenizer_io.py`. Save both tokenizers to `models/…/tokenizer.json`, reload them, encode and decode the 20 sentences before and after, compare ids and text (`find_mismatches`). Inspect `tokenizer.json`. Wrap your tokenizer with `PreTrainedTokenizerFast`, test padding and attention masks, and reload it with `AutoTokenizer.from_pretrained`. Answer the four questions. **Write at least five of your own pytest tests** for your saved tokenizers in `tests/test_my_tokenizers.py` (can be loaded; known text encodes; ids decode; special tokens recognized; encoding consistent after reload).

### Part G — Multilingual benchmark (25 pts) · `src/evaluation.py`, notebook 07
Compare **GPT-2, BERT, FLAN-T5, your BPE and your WordPiece** on `data/eval/eval_sentences.csv` (100 English, 100 French, 50 MSA, 50 Darija).

🧩 Exercises 7.1–7.4 (count tokens through both APIs, compute the metrics by hand with pandas, time with `time.perf_counter`, `pivot` for figures). Implement the adapter that gives both library APIs one interface, the per-sentence results and the summary. Metrics:

| Metric | Definition |
|---|---|
| vocabulary size | \|V\|, including special/added tokens |
| average / maximum tokens per sentence | mean and max of *T<sub>i</sub>* |
| characters per token | Σ characters / Σ tokens (ratio of totals) |
| bytes per token, tokens per word | same idea, in UTF-8 bytes and whitespace words |
| **unknown-token rate** | Σ `[UNK]` / Σ tokens — *read this before any token count* |
| encoding time | mean of repeated timed passes (with warm-up) |
| parity | tokens per word in language *L* ÷ tokens per word in English |

Rules for a fair comparison: same sentences everywhere; **no special tokens counted** (`add_special_tokens=False`); metrics from totals; train and evaluation data kept separate (a test checks it).

Produce: the sentence-level CSV, the comparison table, a table per language, and at least four figures (average tokens per language, vocabulary sizes, sequence-length distributions, characters per token), then answer the analysis questions in the notebook.

### Part H — Report and final challenge (10 pts)
`report.pdf`, 8–10 pages following `report/REPORT_TEMPLATE.md`, with **your** tables and figures.

**Final challenge (1–2 pages inside the report).** You must prepare the tokenizer of a multilingual model for English, French, Modern Standard Arabic and Algerian Darija with a **32 000-token** vocabulary. You do not train it; you design the experiment:

1. Which implementation would you choose (byte-level BPE, Unigram/SentencePiece, WordPiece…) and why — what happens to scripts unseen in training?
2. How would you distribute training data across languages? (e.g. sampling proportional to size vs. temperature-smoothed sampling vs. equal shares)
3. How would you prevent English from taking most of the vocabulary?
4. How would you measure fairness across languages? (use parity, unknown-token rate, characters/bytes per token; say which reference language you choose and why that choice matters)
5. What trade-offs exist between vocabulary size and sequence length? (think about what you saw in Part G)

---

## 7. Suggested plan (≈ 10 h)

| Step | Time |
|---|---:|
| Setup, read this file, notebook 01 (+ Part A answers) | 1.5 h |
| Notebooks 02–05 | 2 h |
| `train_bpe`, `train_wordpiece` + tests | 1.5 h |
| `tokenizer_io` + tests + your own tests | 1 h |
| `evaluation` + tests | 1.5 h |
| Notebook 06, 07 | 1 h |
| Report and final challenge | 1.5 h |

## 8. Common pitfalls

* **Comparing apples and oranges:** BERT and T5 add `[CLS]`/`[SEP]`/`</s>` on their own; count tokens with `add_special_tokens=False`.
* **Trusting token counts:** a WordPiece tokenizer that maps an Arabic word to one `[UNK]` has *very few tokens*. Always report the unknown-token rate.
* **`vocab_size` vs `len(tokenizer)`:** added tokens are not counted by `vocab_size` in Transformers tokenizers.
* **Forgetting the decoder:** without it, byte-level tokens decode to mojibake and WordPiece pieces are not glued back.
* **Forgetting `initial_alphabet=ByteLevel.alphabet()`:** then some bytes may be missing from the vocabulary.
* **Training on the evaluation set**, or reporting numbers from a different run than your saved tokenizer. Commit your trained `tokenizer.json` files and evaluate *those*.
* **Assuming training is reproducible:** WordPiece training in `tokenizers` is not bit-for-bit reproducible (notebook 06 shows it). Save the tokenizer.
* **Timing noise:** one run of a 300-sentence loop is meaningless; use warm-up and repetitions, and say on which machine you measured.

## 9. Additional challenges (bonus, mark them clearly)

1. Train a byte-level BPE on a multilingual corpus (add Arabic/Darija data with `--extra`; check licences) and redo the benchmark.
2. Compare vocabulary sizes 1 000, 2 000, 5 000, 10 000.
3. Study the effect of `min_frequency` on vocabulary construction.
4. Train a Darija-specific tokenizer; compare it with GPT-2 and FLAN-T5.
5. Compare Unicode character count and UTF-8 byte count per language and relate them to byte-level token counts.
6. Explore offset mappings (`return_offsets_mapping=True`) of fast tokenizers.

## 10. References

* Hugging Face `tokenizers`: https://huggingface.co/docs/tokenizers/
* Transformers tokenizer API: https://huggingface.co/docs/transformers/main_classes/tokenizer
* Hugging Face Datasets: https://huggingface.co/docs/datasets/
* Hugging Face LLM Course, chapter 6: https://huggingface.co/learn/llm-course
* Sennrich, Haddow, Birch (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL.
* Kudo (2018). *Subword Regularization.* ACL. · Kudo, Richardson (2018). *SentencePiece.* EMNLP (demo).
* Koehn (2005). *Europarl: A Parallel Corpus for Statistical Machine Translation.* MT Summit.
