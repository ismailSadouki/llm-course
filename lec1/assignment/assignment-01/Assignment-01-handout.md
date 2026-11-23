# Assignment 01 — Text Preprocessing and Tokenization
### Large Language Models: Foundations and Engineering

| | |
|---|---|
| **Type** | Programming + short written analysis + small experiments |
| **Difficulty** | Intermediate |
| **Workload** | about 8–12 hours (read the *Suggested plan* below) |
| **Team** | Individual |
| **Deadline** | To be announced |
| **Submission** | Your GitHub repository (this repo, completed) + `report.pdf` |
| **Needs** | Python 3.10+, a CPU. No GPU, no deep-learning library. |

---

## 1. Why this assignment?

A language model never sees text. It sees **integers**. This assignment builds, by hand, the front of that pipeline:

```
raw text ─► cleaning ─► normalization ─► tokenization ─► token IDs ─► embeddings
 (Part A)    (Part A)      (Part A)       (Parts B, C, D)              (Part F)
```

You will write a conservative cleaning pipeline, three tokenizers (character, word, **BPE from scratch**), measure how they behave on a real corpus, and connect token IDs to the embedding matrix of a Transformer.

The central question of the course for this week:

> **How does the choice of tokenizer change the representation of a text and the cost of processing it?**

> **Scope.** Training tokenizers with the Hugging Face library, studying pretrained tokenizers (GPT-2, BERT, T5) and the multilingual benchmark (English / French / Arabic / Darija) are **Assignment 02**. Here you build the fundamentals yourself; there you will compare them with industrial implementations.

## 2. Learning objectives

By the end you can:

1. contrast classic NLP preprocessing with LLM data preparation, and justify a *conservative* policy;
2. explain Unicode, UTF-8 and NFC/NFD, and apply a documented normalization policy;
3. implement character-level and word-level tokenizers with special tokens and unknown-token handling;
4. **implement Byte Pair Encoding from scratch** (training and encoding) and explain every step;
5. measure vocabulary size, sequence length and out-of-vocabulary behaviour on real text;
6. explain how WordPiece and Unigram differ from BPE;
7. explain how token IDs become vectors and how vocabulary size drives the size of the embedding matrix.

## 3. Setup

```bash
git clone <your-repo-url> assignment-01 && cd assignment-01
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                      # 'red' everywhere: that is the expected starting point
jupyter notebook            # then open notebooks/01_preprocessing.ipynb
```

**Libraries.** Allowed: the Python standard library, NumPy, Pandas, Matplotlib. **Not allowed in this assignment:** `tokenizers`, `transformers`, `tiktoken`, `sentencepiece`, NLTK/spaCy tokenizers, or any code that already implements BPE. (They come back in Assignment 02.)

## 4. How to work: exercises first, then tests, part by part

The assignment has **two kinds of programming work**, and they feed each other:

| | What | Where | Feedback |
|---|---|---|---|
| 🧩 **Code-completion exercises** (21) | Fill the `____` blanks of a small, self-contained piece of the algorithm: Unicode normalization, a whitespace regex, an `HTMLParser` subclass, a vocabulary, `Counter` counting, pair counting with `zip`, the merge loop, the BPE tie rule, NumPy lookups… Every exercise has **hints** (the Python functions to use) | the notebooks | a ✅ *check* cell right after each exercise |
| 🛠 **Functions in `src/`** | Package your working pieces into reusable, tested functions (each docstring says which exercise it comes from) | `src/*.py` | `pytest` |

Typical loop for a part: **do the 🧩 exercises → move the working code into `src/` → `pytest` → finish the notebook (experiments + written answers).**

**How a 🧩 exercise works.** You see something like

```python
pairs = Counter()
for symbols, freq in table.items():
    for a, b in ____:
        pairs[(a, b)] ____
```
Read the hints above the cell, replace each `____`, run the cell, then run the ✅ check cell. If the check fails, the assertion tells you what is wrong. Do not delete the check cells, and You can use AI but it's better to not look for a solution online: the hints and the Python documentation are enough.

Every function you must write in `src/` is a stub that raises `NotImplementedError`, with a docstring that tells you exactly what is expected. The tests in `tests/` are the specification:

```bash
pytest tests/test_preprocessing.py     # Part A
pytest tests/test_tokenizers.py        # Part B  (use -k Character or -k Word)
pytest tests/test_bpe.py -x            # Part C  (-x: stop at first failure, go top to bottom)
pytest tests/test_embeddings.py        # Part F
pytest                                 # everything
```

Read the failing test when something is red: it usually tells you the case you forgot. Passing the tests is necessary, not sufficient: the notebooks and the report are where you show that you understand.

Commit after every green step (`git commit -m "Part A: remove_html passes"`). Small commits are the best backup and the best evidence of your own work.

### Repository map

| Path | Status | Content |
|---|---|---|
| `src/preprocessing.py` | **TODO** | cleaning pipeline (the classic-NLP baseline `aggressive_normalize` is provided) |
| `src/char_tokenizer.py` | **TODO** | `CharacterTokenizer` (constructor provided) |
| `src/word_tokenizer.py` | **TODO** | `WordTokenizer` (regex convention and `detokenize` provided) |
| `src/bpe.py` | **TODO** | `BPETokenizer` — the core of the assignment |
| `src/embeddings.py` | **TODO** | NumPy embedding layer |
| `src/evaluation.py` | provided | split + metrics (tokens/word, characters/token, unseen words) |
| `tests/` | provided | the specification; **do not edit** (you may add your own tests in new files) |
| `notebooks/01…05` | **TODO** | 21 🧩 code-completion exercises (hints + check cells) + guided experiments + your written answers |
| `docs/WALKTHROUGH.md` | provided | explanatory guide to every part (read it first) |
| `data/toy`, `data/raw`, `data/real` | provided | three datasets, see `data/README.md` |
| `report/REPORT_TEMPLATE.md` | provided | structure of the report |
| `results/` | you | CSV tables and figures produced by your notebooks |

## 5. Grading

| Part | Topic | Pts | Assessed by |
|---|---|---:|---|
| A | Preprocessing | 15 | tests (8) + exercises 1.1–1.5 (2) + notebook 01 analysis (5) |
| B | Character and word tokenizers | 15 | tests (8) + exercises 2.1–2.4 (2) + notebook 02 experiments & answers (5) |
| C | **BPE from scratch** (toy corpus) | 30 | tests (18) + exercises 3.1–3.6 (3) + notebook 03 trace, experiments & answers (9) |
| D | BPE on real data | 15 | notebook 04: exercises 4.1–4.3 (3) + results, figures, interpretation (12) |
| E | WordPiece and Unigram (written) | 10 | report section |
| F | Embeddings | 10 | tests (4) + exercises 5.1–5.3 (2) + notebook 05 answers (4) |
| G | Report, reproducibility, code quality | 5 | report + repo |
| | **Total** | **100** | |

**Rules**
* Code that does not run, or notebooks that cannot be re-executed top to bottom, lose points.
* The BPE learning algorithm must be **your own**. 
* Every number in the report must come from your code. Never invent a measurement.


---

## 6. The tasks

### Part A — Preprocessing for LLM pipelines (15 pts)
🧩 *Exercises 1.1–1.5:* Unicode (`unicodedata`), whitespace (`re.sub`), HTML (complete an `HTMLParser` subclass), order-preserving deduplication (`set`), collisions (`pandas.groupby`).

**Files:** `src/preprocessing.py` · `notebooks/01_preprocessing.ipynb` · **Tests:** `tests/test_preprocessing.py`

Traditional NLP lowercases, stems and removes stop words. Modern LLM pipelines clean the data **without destroying linguistic information**. Implement the conservative pipeline:

| Step | Function | Policy |
|---|---|---|
| HTML → text | `remove_html` | keep visible text; decode entities; drop `<script>`/`<style>` content |
| Unicode | `normalize_unicode` | **NFC** |
| Whitespace | `normalize_whitespace` | collapse runs to a single space, strip ends |
| One document | `preprocess_text` | `None`/non-string → `""`; idempotent |
| Corpus | `drop_empty`, `remove_duplicates`, `preprocess_corpus` | exact duplicates, order kept, statistics reported |

Never touch punctuation, capitalization, stop words or word order.

In the notebook you will (1) answer eight theory questions (stop words, Unicode vs UTF-8, NFC/NFD, deduplication, multilingual pitfalls…), (2) run your pipeline on the messy dataset, and (3) compare *original / aggressive / conservative* versions of six sentences to see which different meanings collapse into the same string.

### Part B — Basic tokenization (15 pts)
🧩 *Exercises 2.1–2.4:* a character vocabulary and encode/decode, counting and ordering words with `Counter`, the OOV rate.

**Files:** `src/char_tokenizer.py`, `src/word_tokenizer.py` · `notebooks/02_basic_tokenizers.ipynb` · **Tests:** `tests/test_tokenizers.py`

Implement `CharacterTokenizer` and `WordTokenizer` with the special tokens `[PAD]=0 [UNK]=1 [BOS]=2 [EOS]=3`, deterministic vocabularies, `encode`/`decode`, unknown-token handling, and (for words) `min_freq`, `max_vocab_size` and `oov_rate`.

In the notebook, on the real corpus, you will compare vocabulary sizes and sequence lengths, and plot the OOV rate on held-out text against vocabulary size.

### Part C — Byte Pair Encoding from scratch (30 pts)
🧩 *Exercises 3.1–3.6 walk you through the algorithm on the toy corpus (word table, weighted pair counts, tie rule, merge, training loop, encoding a new word); you then package them in `src/bpe.py`.*

**Files:** `src/bpe.py` · `notebooks/03_bpe_toy.ipynb` · **Tests:** `tests/test_bpe.py`

BPE starts from characters and repeatedly merges the most frequent adjacent pair into a new symbol. Each word is a tuple of characters plus an end-of-word marker `</w>`, weighted by its frequency:

```
low   (x3) -> l o w </w>
lower (x2) -> l o w e r </w>
```

You implement the algorithm as five small functions (the docstrings contain the details and one fully worked example):

1. `initialize_vocabulary` — corpus → word-frequency table
2. `get_pair_frequencies` — weighted adjacent-pair counts
3. `merge_pair` — replace a pair everywhere (left to right, no mutation)
4. `train` — the loop; **tie rule:** highest frequency, then lexicographically smallest pair
5. `encode_word` / `encode` / `decode` — apply the learned rules **in the order they were learned**

Work through `pytest tests/test_bpe.py -x` from top to bottom. Then, in notebook 03:

* **trace** the algorithm step by step and check the first two merges **by hand on paper**;
* train **BPE-10, BPE-30, BPE-50**, record merge rules, vocabulary size, tokens per word, behaviour on unseen words, and **explain** why BPE-30 and BPE-50 turn out identical;
* answer the eight theory questions of C.5 (why merge frequent pairs, why frequencies, token vs merge rule, why order matters, unseen words, character-level vs byte-level BPE…).

### Part D — BPE on real data (15 pts)
🧩 *Exercises 4.1–4.3:* fertility and characters per token by hand, unseen-word rate with sets, token-length histogram.

**Notebook:** `notebooks/04_bpe_real_data.ipynb` (no new code to write beyond a few cells)

A toy corpus proves that your code is correct; a real corpus shows what BPE *does*. Train on the training split of Tiny Shakespeare with 100, 500 and 1000 merges, then evaluate **on held-out lines**: tokens per word, characters per token, training/encoding time, what the first and later merges look like, and how unseen words are segmented. Interpret your results (diminishing returns, corpus artifacts, comparison with the character and word tokenizers of Part B). Optional extensions (punctuation handling, speed-up) are described in the notebook.

### Part E — WordPiece and Unigram (10 pts, written)
You do **not** implement these algorithms. Study them and write about one page in your report (with citations):

*WordPiece* — (1) how is its vocabulary built? (2) how does its merge criterion differ from BPE's raw pair frequency? (3) what does the `##` prefix mean? (4) why does BERT use it?
*Unigram LM* — (1) how is the initial vocabulary built? (2) how are probabilities assigned to subwords? (3) how does pruning work? (4) how does segmentation differ from BPE (hint: BPE is deterministic given the merge list; Unigram searches for the best segmentation)?

Then complete this table:

| Property | BPE | WordPiece | Unigram |
|---|---|---|---|
| Main idea | | | |
| Vocabulary learning (bottom-up / top-down?) | | | |
| Segmentation at inference | | | |
| Probabilistic model? | | | |
| Example model that uses it | | | |

### Part F — From token IDs to embeddings (10 pts)
🧩 *Exercises 5.1–5.3:* NumPy embedding matrix, lookup by fancy indexing, one-hot × matrix, batches.

**Files:** `src/embeddings.py` · `notebooks/05_embeddings.ipynb` · **Tests:** `tests/test_embeddings.py`

Tokenization gives integers; an embedding matrix `E ∈ ℝ^{V×d}` turns them into vectors. Implement the lookup layer in NumPy, show that lookup equals *one-hot × E*, and answer: for V = 10 000 and d = 256 — what is the shape and parameter count of E? What are the shapes of a 12-token id sequence and of its embedding? How does the lookup work, and why is E trainable? Finally, connect it to your own BPE vocabulary and discuss the trade-off *large vocabulary (big E) vs. long sequences*.

### Part G — Report and repository (5 pts)
`report.pdf`, **6–8 pages**, following `report/REPORT_TEMPLATE.md`: introduction, preprocessing, tokenizers, BPE (algorithm + toy + real), WordPiece/Unigram, embeddings, limitations, references, reproducibility appendix. Include **actual** tables and figures from `results/`. Your repo must contain `requirements.txt`, a README line saying how to reproduce everything, and clean, commented code.

---

## 7. Suggested plan (≈ 10 h)

| Step | Time | Done when |
|---|---:|---|
| Read this file, set up, run `pytest` once | 0.5 h | environment works |
| Part A code + notebook 01 | 2 h | `test_preprocessing.py` green |
| Part B code + notebook 02 | 1.5 h | `test_tokenizers.py` green |
| Part C code (`get_pair_frequencies` → `train` → `encode_word`) | 2.5 h | `test_bpe.py` green |
| Notebook 03 (trace by hand, BPE-10/30/50) | 1 h | table + answers |
| Notebook 04 (real data) | 1 h | results + figures (training runs 1–3 min) |
| Part F + notebook 05 | 0.5 h | tests green |
| Part E reading + report | 1.5 h | `report.pdf` |

## 8. Common pitfalls

* **Regex-only HTML cleaning** — `re.sub(r"<[^>]+>", " ", x)` leaves "very ." instead of "very." and can eat `5 < 10 > 3`. Use a parser and think about inline vs block tags.
* **Unescaping entities too early** — `&lt;` becomes `<` and then looks like a tag. Order matters.
* **Merging overlapping pairs** — `a a a` with merge `(a, a)` must give `aa a`.
* **Forgetting `</w>`** — then `er` inside `newer` and at the end of `lower` look identical. The marker is what distinguishes word endings.
* **Unweighted pair counts** — count each word *times its frequency*.
* **Applying merges out of order** — the rule learned first must be applied first.
* **Slow `encode_word`** — looping over all 1000 merges for every word is slow; use the rank-based approach described in the docstring.
* **Evaluating on training data** — report metrics on the held-out split.
* **Reading numbers off the notebook that are not from the final run** — re-run top to bottom before you write the report.

## 9. What comes next (Assignment 02)

You will train a production-grade BPE tokenizer with Hugging Face `tokenizers`, inspect the pretrained tokenizers of GPT-2, BERT and T5, and benchmark them on English, French, Arabic and Algerian Darija. Your from-scratch BPE gives you the vocabulary to understand every line of that comparison.

## 10. References

1. Sennrich, Haddow, Birch (2016). *Neural Machine Translation of Rare Words with Subword Units.* ACL.
2. Schuster, Nakajima (2012). *Japanese and Korean Voice Search.* ICASSP.
3. Wu et al. (2016). *Google's Neural Machine Translation System.* (WordPiece in practice)
4. Kudo (2018). *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates.* ACL.
5. Kudo, Richardson (2018). *SentencePiece.* EMNLP (system demonstrations).
6. Devlin et al. (2019). *BERT.* NAACL.
7. The Unicode Standard — Normalization Forms (UAX #15): https://unicode.org/reports/tr15/
8. Hugging Face LLM Course, chapter 6 (tokenizers): https://huggingface.co/learn/llm-course
