# Assignment 01 — Walkthrough Guide
### Text Preprocessing and Tokenization (from scratch)

> **What this guide is.** A guided tour of the assignment: the ideas behind each part, what to notice, how to approach each 🧩 exercise and each function, and the mistakes students usually make.
> **What it is not.** It does not contain the answers to the graded questions or to the exercises. Demonstrations use *other* examples than the ones in your notebooks, on purpose.

Read it once before you start; keep it open next to the handout (`README.md`).

---

## Contents

0. The big picture
1. How the work is organised (🧩 exercises, `src/`, tests)
2. Part A — Preprocessing
3. Part B — Character and word tokenizers
4. Part C — BPE from scratch (the core)
5. Part D — BPE on real data
6. Part E — WordPiece and Unigram (written)
7. Part F — Token ids and embeddings
8. Report
9. Debugging cheat sheet
10. Self-check before submitting

---

## 0. The big picture

### 0.1 The pipeline

A language model never sees text; it sees integers. This assignment builds the front of that pipeline by hand:

```
raw text ──► cleaning ──► normalization ──► tokenization ──► token ids ──► embeddings
 (Part A)     (A)           (A)             (Parts B, C, D)                (Part F)
```

| Stage | Question it answers | Typical operations |
|---|---|---|
| cleaning | Is this record usable at all? | drop markup, drop empty/missing records, drop duplicates |
| normalization | Which different encodings of the same content should be one? | Unicode normalization, whitespace |
| tokenization | Which units will the model see? | characters, words, **subwords (BPE)** |
| ids | How do units become numbers? | a vocabulary: unit ↔ integer |
| embeddings | How do numbers become vectors? | a trainable matrix with one row per id |

### 0.2 Two ideas to keep in mind all assignment long

1. **Information is easy to destroy and impossible to recover.** Each step in Part A and each tokenizer in Parts B–D either *preserves* information (you can go back) or *loses* it. For every operation you write, ask: "can I undo this?"
2. **Every tokenizer is a trade-off between vocabulary size and sequence length.** Small vocabulary → long sequences, big vocabulary → short sequences and rare, poorly-learned units. Characters, words and BPE sit at different points of that trade-off; Parts B, D and F let you measure it.

### 0.3 Vocabulary of the assignment

* **token** — one unit produced by a tokenizer (a character, a word, a subword).
* **id** — the integer index of a token in the vocabulary. An arbitrary label.
* **vocabulary** — the table token ↔ id.
* **special token** — a control symbol (`[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`) that is not part of the text.
* **OOV (out of vocabulary)** — an input unit the tokenizer has no entry for.

---

## 1. How the work is organised

### 1.1 Two kinds of programming work

| | Where | What you do | Feedback |
|---|---|---|---|
| 🧩 **exercises** (21) | notebooks | fill the `____` blanks of one small piece of an algorithm | a ✅ check cell |
| 🛠 **functions** | `src/*.py` | package your working pieces, make them robust | `pytest` |

The exercises are *deliberately small*: each isolates one idea (e.g. "count neighbouring pairs", "merge left to right"). The `src/` functions combine them and must handle every edge case. The docstring of each `src/` function says which exercise it comes from.

### 1.2 How to solve a 🧩 exercise

1. **Read the statement and the hints first.** Hints name the Python functions you need.
2. **Look at the code around the blank.** What *type* must it be (a function call? a number? a tuple?). The left-hand side of an assignment and the names used afterwards tell you.
3. **Experiment in a scratch cell** before filling the blank: `help(zip)`, `print(type(x))`, a tiny example by hand.
4. **Run the cell, then the ✅ check.** If it fails, read the assertion: it states what was expected.
5. **Explain your line in one sentence** to yourself. If you can't, you copied; redo it.
6. **Then move it into `src/`.** Replace the constants of the exercise by function parameters.

Reading error messages is part of the skill:

| Message | Usually means |
|---|---|
| `NameError: name '____' is not defined` | a blank is still there |
| `TypeError: ... takes ... arguments` | wrong number/type of arguments — check `help()` |
| `KeyError` | you looked up something that is not in the dict — use `.get` or check your keys |
| `AssertionError` in a ✅ cell | your code runs but gives a wrong result — compare with what the check expects |

### 1.3 Reading a failing test

```bash
pytest tests/test_bpe.py -x        # stop at the FIRST failure
```

1. Read the **test name** — it describes a behaviour (`test_overlapping_pairs_are_merged_left_to_right`).
2. Look at the **assert line** — expected vs actual.
3. Reproduce the case in a notebook cell, `print` intermediate values.
4. Fix; re-run **the same file**; move to the next failure.

Work through `test_bpe.py` from the top: the classes are ordered like the algorithm.

### 1.4 Habit: predict, run, explain

For every experiment write one sentence *before* running it: "I expect X because Y." Then run it. When reality disagrees with you, you have found what the report should explain.

---

## 2. Part A — Preprocessing

**Goal:** a cleaning pipeline that removes junk but keeps everything that carries linguistic information (punctuation, capitalization, stop words, word order).

### 2.1 The mindset change

Classic NLP pipelines shrink the data to make a small model's life easier (lowercase, no punctuation, no stop words, stems). A modern LLM is trained to *predict the next token of real text*: it needs the real text. So the question for every transformation is not "does it simplify?" but "does it destroy something the model would have to produce?".

### 2.2 Unicode (exercise 1.1)

Characters on screen are not characters in memory. One visible `å` can be one code point or two (a base letter plus a combining ring):

```python
import unicodedata
[hex(ord(c)) for c in "å"]                                   # ['0xe5']            one code point
[hex(ord(c)) for c in unicodedata.normalize("NFD", "å")]     # ['0x61', '0x30a']  'a' + combining ring
```

Two strings that *look* identical can compare unequal, hash differently, tokenize differently, and fail to be recognised as duplicates. Normalization picks one canonical representation.

Things to investigate yourself (the notebook has the cells):

* the difference between **code points** and **bytes** (`len("é")` vs `len("é".encode("utf-8"))`);
* what **NFC** and **NFD** do, and what the *K* in **NFKC** adds — run both on the ligature and the Arabic example of the notebook and compare;
* what happens to Arabic diacritics, tatweel and presentation forms under each form.

This is where you decide the policy (and must justify it).

### 2.3 Whitespace (exercise 1.2)

A regular expression is a tiny language for describing sets of strings. Learn three pieces: a character class (`\s`), a quantifier (`+`), and substitution (`re.sub`). Before writing the pattern, list which characters count as "whitespace" in your data (space, tab, newline, non-breaking space…). Decide, and write in your report, what you do with paragraph breaks: collapsing newlines loses structure — is that acceptable for this corpus?

### 2.4 HTML (exercise 1.3) — why a parser, not a regex

Try this *in your scratch cell* before the exercise:

```python
import re
re.sub(r"<[^>]+>", " ", "This is <b>very</b>. Really.")     # 'This is  very . Really.'
```

The space inserted for `<b>` and `</b>` detaches the full stop from `very`. Inline tags (`<b>`, `<i>`, `<a>`) must not introduce a space; block tags (`<p>`, `<div>`, `<br>`, `<li>`) must. A regex cannot know which is which, cannot skip the *content* of `<script>`, and gets confused by escaped text such as `5 &lt; 10`.

`html.parser.HTMLParser` reads the document and **calls your methods** when it meets a start tag, an end tag or text — an *event-driven* design you will meet again (SAX parsers, tokenizers, state machines). The exercise gives you the skeleton: decide in each method what to remember (a depth counter, a list of text pieces).

Think about **order**: if you decode entities (`&lt;` → `<`) *before* removing tags, what could go wrong?

### 2.5 Deduplication (exercise 1.4)

"Exact duplicate" means identical strings. Two design questions to answer in your report:

* Compare **before** or **after** cleaning? What would each choice treat as duplicates (try two documents that differ only by whitespace; two that differ only by NFC/NFD)?
* Is `"Apple"` a duplicate of `"apple"`? Why does your pipeline say what it says?

Implementation idea: a `set` of what you have already seen gives O(1) membership. A set has no order — so how do you *keep* order?

### 2.6 Putting it together: `preprocess_text`, `preprocess_corpus`

* What should a missing record (`None`) become? What about a number that sneaked into a text column?
* Which order of steps (HTML → Unicode → whitespace, or another) gives correct results? Find a concrete input where your order matters.
* **Idempotence:** applying the pipeline twice must give the same result as once. Why is this a desirable property for a data pipeline?
* The `stats` dictionary is your audit trail: in real pipelines the first thing you check is *how much data each step removed*.

### 2.7 The experiment (A.3)

You compare *original / aggressive / conservative* versions of six sentences. Don't just look at the table: ask for each pair, **"if a model saw only the processed version, could it tell the two sentences apart? could it generate the original?"**. Exercise 1.5 (`groupby`) automates the "did two sentences collide?" question; the written discussion is about *why it matters*.

The bonus cell (aggressive pipeline applied to NFD text) shows something about the order of operations that is worth a sentence in your report.

---

## 3. Part B — Character and word tokenizers

**Goal:** two tokenizers at the two extremes of the trade-off, with proper special-token and unknown-token handling.

### 3.1 What a tokenizer object does

```
train(texts)  →  builds token_to_id / id_to_token
encode(text)  →  list of ids
decode(ids)   →  text
```

Everything else (special tokens, unknown handling, determinism) is detail you must get right. Note the **separation of concerns**: *training* decides the vocabulary; *encoding* only uses it. Encoding must never change the vocabulary.

### 3.2 Special tokens

`[PAD] [UNK] [BOS] [EOS]` take ids 0–3, *before* any regular token. Why does it help to fix their ids once and for all? (Think about padding a batch and about two tokenizers retrained on different data.)

Decide and document: when decoding, should special tokens appear in the text? The tests specify a `skip_special_tokens` behaviour; understand what it implies for `[UNK]`.

### 3.3 Determinism (exercises 2.1, 2.3)

A vocabulary built from a `set` has an arbitrary order that can change between runs. **Reproducible ids** need a defined order: sorted characters; for words, decreasing frequency with an alphabetical tie-break. The tuple trick of exercise 2.3 — *a tuple as a sort key, negated number for descending order* — is one you will reuse (BPE's tie rule is the same idea).

### 3.4 Unknown characters/words (exercise 2.2)

When the input contains a unit the vocabulary lacks, you must emit *something*: that is the `[UNK]` id. `dict.get(key, default)` expresses it in one call. After doing it, think about what is **lost**: can you decode back to the original? What would a tokenizer need to *never* produce `[UNK]`? (Keep that question — Part C question 8 and Assignment 02 build on it.)

### 3.5 The word tokenizer's convention

The regular expression that defines a "word" is provided; read it slowly and test it on awkward inputs (apostrophes, French elisions, punctuation, digits, Arabic). Whatever it does is the **convention** you must document: it decides what counts as one token, and therefore what OOV means.

`decode` cannot be perfectly lossless for word-level tokens because whitespace was discarded. Find the cases where it is not, before the notebook asks you to.

### 3.6 OOV rate (exercise 2.4)

Be precise about *what* you count: occurrences or distinct words? Training text or unseen text? The notebook sweeps `min_freq` and plots the OOV rate on training vs held-out text — before running it, **predict the shape of both curves** and where they will differ.

### 3.7 Questions to prepare

The graded questions compare vocabulary size, sequence length, robustness and fit for morphologically rich languages. Gather *numbers* from the real-corpus cells (vocabulary sizes, tokens on the test set, OOV) rather than answering from general principles.

---

## 4. Part C — BPE from scratch

This is 30 % of the grade and the heart of the assignment. Slow down: the algorithm is short, but each step has a subtlety that the tests check.

### 4.1 The idea in one paragraph

Start with every word split into characters. Find the pair of neighbouring symbols that occurs most often in the whole corpus, **glue** it into a new symbol, and repeat. Each glue operation is a *merge rule*; the ordered list of merge rules **is** the tokenizer. To tokenize a new word, split it into characters and replay the merge rules in the order they were learned.

### 4.2 A full trace on a *different* corpus (not yours)

Corpus: `hug hug hug pug pug pun pun bun`. Each word gets an end-of-word marker `</w>` and keeps its frequency:

```
3 x h u g </w>
2 x p u g </w>
2 x p u n </w>
1 x b u n </w>
```

Weighted pair counts (top of the list):

```
(u,g) 5    (g,</w>) 5    (p,u) 4    (h,u) 3    (u,n) 3    (n,</w>) 3
```

`(u,g)` occurs in `hug` (×3) and `pug` (×2): 3 + 2 = 5 — **each occurrence counts as often as the word**. `(u,g)` and `(g,</w>)` tie at 5. With the rule "highest frequency, then lexicographically smallest pair", `('g','</w>')` wins: `"g" < "u"`. After 6 merges (this is real output of a reference implementation):

```
merges:      (g,</w>) (u,g</w>) (h,ug</w>) (n,</w>) (u,n</w>) (p,ug</w>)
frequencies:    5        5         3          3        3          2
```

Notice: the **frequencies of the chosen pairs never increase** (can you see why a freshly created symbol cannot be more frequent than the pair that created it?). Encoding the *unseen* word `hugs pun mug` with these merges gives pieces like `h u g s </w>`, `p un</w>`, `m ug</w>`: parts of words that were never seen whole.

Reproduce a trace like this by hand on **your** toy corpus — the notebook requires it, and nothing builds understanding faster.

### 4.3 The six pieces (exercises 3.1 – 3.6)

| # | Function | One-sentence job | Subtlety |
|---|---|---|---|
| 3.1 | word table | turn text into `{(symbols…): frequency}` | the end marker `</w>`; tuple, not list (why must a dict key be hashable?) |
| 3.2 | pair counts | count neighbouring pairs, weighted | a pair twice in a word counts twice; `zip(s, s[1:])` |
| 3.3 | best pair | choose by frequency, tie → smallest pair | tuple key `(-freq, pair)` |
| 3.4 | merge | replace a pair everywhere in every word | scan **left to right**, advance by 2 after a merge; never mutate the input; add frequencies if two words become identical |
| 3.5 | training loop | count → choose → merge, record the rule | stop when there is no pair left |
| 3.6 | encode a word | replay the rules in learned order | rank-based: always apply the *earliest-learned* applicable rule |

**Why the end-of-word marker?** Without it, what would distinguish the `er` in `newer` from the `er` at the *start* of `ermine`? Test your own answer in a scratch cell.

**Why left to right on overlapping pairs?** Merge `(a, a)` in `a a a`. Both `(aa, a)` and `(a, aa)` are conceivable; the tests fix one. Decide which a left-to-right scan produces.

**Why does the order of merges matter in 3.6?** Build a small example with two merge rules where applying them in the opposite order gives a different segmentation (the test `test_merge_order_matters` is a hint). This is a graded theory question: have *your own* example ready.

**Complexity (you will need this in Part D).** The naive `train` recomputes all pair counts after every merge. For `V` unique words of average length `L` and `M` merges, that is roughly `M × V × L` operations. On the toy corpus it is instantaneous; on a real corpus with tens of thousands of unique words it takes a minute — which is fine, but be aware, and avoid doing needless work inside the loop (for example re-splitting strings).

### 4.4 Attributes the tests expect

`merges` (ordered list of pairs), `merge_frequencies` (the count at each step), `merge_ranks` (`{pair: index}`), `word_freqs` (the table after the last merge), `token_to_id`/`id_to_token` (`[UNK]` = 0, then the sorted base symbols, then one id per merge in learning order). The stub's docstring defines each; the tests assert them.

### 4.5 Packaging checklist (from exercise to `src/`)

- [ ] parameters instead of constants (`pair`, `table`, `num_merges`);
- [ ] no mutation of inputs;
- [ ] `train` can be called twice on the same object (reset state);
- [ ] works with a string *or* a list of lines;
- [ ] unseen characters survive in `encode_word` (they stay single symbols);
- [ ] `decode(encode(x)) == x` for whitespace-normalised text.

### 4.6 The experiments (BPE-10 / 30 / 50)

Predict before running: how many merges will each tokenizer really learn on this corpus? What will the table look like for BPE-30 and BPE-50? (If your answer is "the same", say why.) When you inspect unseen words and the unseen character `z`, ask *what is guaranteed* about BPE's ability to represent text, and what is not.

### 4.7 Theory questions (C.5): how to prepare

For each of the eight questions, find **a concrete example in your own runs** and write the answer around it. Answers that only paraphrase a textbook score poorly; answers that point at "in my BPE-10 run, the word `…` was split as … because the merge `…` had rank …" score well.

---

## 5. Part D — BPE on real data

**Goal:** see what BPE does at scale, and learn to evaluate a tokenizer honestly.

### 5.1 Why a second corpus?

The toy corpus proves your code is correct. It says nothing about behaviour: nine words cannot show diminishing returns, rare words, or corpus quirks. Tiny Shakespeare (≈ 1 MB of plays) can.

### 5.2 Train/test discipline

Train on **train lines**, evaluate on **held-out test lines** (`split_lines`, seed 0). Measuring on the data the tokenizer was trained on is flattering (every word is "known"). Exercise 4.2 makes the difference visible: how many test words did the training text never contain?

Also note the corpus is **not deduplicated** here on purpose: repeated speaker names are genuine data. Be ready to explain why this choice differs from the preprocessing policy of Part A.

### 5.3 Metrics — what they mean

| Metric | Reads as | Better when |
|---|---|---|
| vocabulary size | number of entries (incl. `[UNK]`) | — (it's a cost: bigger embedding table) |
| tokens per word ("fertility") | how many pieces a word is cut into | lower |
| characters per token ("compression") | how much text one token covers | higher |
| unseen-word rate | fraction of test words never seen in training | — (a property of the data) |

Exercise 4.1 asks you to recompute fertility and compression with plain loops so that you know exactly what they count (totals over all lines, not an average of per-line ratios — think about why).

### 5.4 Reading the results

* **Curves:** plot the metric against the number of merges. Where is the curve steep, where flat? What does "diminishing returns" mean in terms of *what the extra merges are learning*?
* **Merges:** look at the first 25 and at merges 500–515. What kind of pair is learned early (very frequent, short) and what kind later (rarer, longer)? Does what you see depend on this *particular* corpus? (Look at the longest tokens: do any look like corpus artefacts rather than language?)
* **Unseen words:** how are they segmented at 100 vs 1000 merges? What does that say about BPE's treatment of the OOV problem you measured in Part B?
* **Token lengths (exercise 4.3):** the histogram of vocabulary token lengths tells you what mix of short and long units the budget bought.
* **Time:** training time grows with the number of merges; report it, and note what you would optimise (the optional extensions describe ideas).

### 5.5 Optional extensions

Punctuation handling, incremental pair counts (speed), lower-casing. If you do one, treat it as a controlled experiment: change one thing, keep everything else fixed, report before/after on the **same test lines**.

---

## 6. Part E — WordPiece and Unigram (written)

No code. This is a short literature-based comparison, written in your own words with citations. How to approach it:

1. Read the Sennrich et al. paper first (you now know BPE). Then the WordPiece description in the Wu et al. / Schuster–Nakajima references and the Unigram part of Kudo (2018).
2. For each algorithm, answer the same four questions: **where does the initial vocabulary come from? how is it grown or shrunk? what criterion chooses what to merge/keep? how is a new word segmented at inference?**
3. For the comparison table, make each cell a *short, falsifiable statement* ("merges the pair with highest …", "segments by …"), not a slogan.
4. Check yourself with one tiny example per algorithm (a three-word corpus you can run by hand).
5. Cite precisely (paper, section). Do not paste text.

---

## 7. Part F — Token ids and embeddings

### 7.1 From id to vector

An embedding layer is **a table**: one row (a vector of `d` numbers) per vocabulary id. "Looking up token 7" means taking row 7. That is all.

Practise shape reasoning with *different numbers* than the assignment (the assignment's numbers are a graded question):

* Table with V = 50 rows and d = 8 columns → shape `(50, 8)`, `50 × 8 = 400` entries, hence 400 trainable parameters.
* A sentence of 5 ids → an id array of shape `(5,)`; its embedding has shape `(5, 8)` (one row per id).
* A batch of 3 such sentences → ids `(3, 5)`; embeddings `(3, 5, 8)`.

The rule: **the output shape is the id shape with `d` appended.** NumPy "fancy indexing" (`E[ids]`) does exactly that — exercise 5.1.

### 7.2 One-hot × matrix (exercise 5.2)

A one-hot vector has a single 1 at position `i`. Multiply it by `E` and what do you get? Do the 2×2 case by hand before using NumPy. This explains how the lookup relates to a **linear layer** and why it is *trainable*: the gradient only flows into the rows that were used. Then ask why real frameworks do the lookup instead of the multiplication.

### 7.3 Connecting to your tokenizer

The embedding table has `V` rows where `V` is **your tokenizer's vocabulary size** — including `[UNK]` and special tokens. In the notebook you connect BPE ids to an embedding table, and then compute the parameter count for your real-data vocabularies. Use these numbers for the closing trade-off question: big vocabulary (big table, short sequences) vs. small vocabulary (small table, long sequences).

### 7.4 Pitfalls

* ids outside `[0, V)` must raise an error, not wrap around (negative indices are valid in NumPy — the tests check you reject them).
* Don't confuse the **shape of E** with the **shape of the output**.
* Random initialisation needs a seed for reproducibility.

---

## 8. The report

Use `report/REPORT_TEMPLATE.md`. Think of it as an argument, not a diary:

* **Claim** (e.g. "more merges shorten sequences with diminishing returns") → **evidence** (your table/figure) → **explanation** (what BPE is doing) → **limits** (one small corpus, one domain, whitespace words…).
* Every number comes from a file in `results/` produced by your notebooks. If you re-run an experiment, re-generate the tables.
* Document your decisions (NFC vs NFKC, order of steps, newline policy, tie rule, how you treat unseen characters).
* Acknowledge external help (including AI assistants) and cite sources.

---

## 9. Debugging cheat sheet

| Symptom | Likely cause | Where to look |
|---|---|---|
| `NotImplementedError` | function not written yet | the stub in the traceback |
| tests pass in the notebook but fail in pytest | notebook code not moved to `src/` | the function named in the failure |
| `remove_html` leaves `very .` or eats `5 < 10` | regex instead of a parser / entities decoded too early | exercise 1.3 |
| ids change from run to run | vocabulary built from an unordered `set` | sort before assigning ids |
| `merge_pair` result changes the input table | in-place modification | build a new dict |
| `aa a` vs `a aa` mismatch | scanning direction / overlapping pairs | exercise 3.4 |
| first merge is not the expected one | counts not weighted, or tie rule missing | exercises 3.2, 3.3 |
| `encode_word` correct but very slow on real data | loops over all merges for every word | use ranks (exercise 3.6) |
| `decode(encode(x)) != x` | whitespace normalisation / unknown characters | what the docstring promises |
| metrics look too good | evaluated on training lines | use `test_lines` |
| notebook ✅ passes but pytest fails | the function in `src/` handles fewer cases than your exercise | read the failing test name |
| edits to `src/` invisible in the notebook | stale import | restart the kernel |

---

## 10. Self-check before submitting

**Code**
- [ ] All 21 🧩 exercises completed (no `____` left) and every ✅ passes.
- [ ] `pytest` is fully green.
- [ ] You did not edit the provided tests (extra tests go in new files).

**Experiments**
- [ ] Notebooks 01–05 re-run top to bottom without error.
- [ ] `results/` holds the tables and figures you cite in the report.
- [ ] Metrics were computed on held-out lines.

**Understanding** — can you explain, without notes:
- [ ] why a conservative pipeline is safer than an aggressive one, with an example;
- [ ] what NFC does and why a decomposed `é` must be normalized before deduplication;
- [ ] each of the six BPE pieces and the tie rule;
- [ ] why merge order matters, with your own example;
- [ ] what an embedding lookup is, and why vocabulary size drives parameters;
- [ ] one limitation of your experiments.

**Report**
- [ ] Every number traceable; decisions documented; references cited.

---

*The goal is not to call a tokenizer; it is to understand how text becomes the input of a language model. Predict, run, explain.*
