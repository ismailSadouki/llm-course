# Assignment 02 — Walkthrough Guide
### Text Processing and Tokenization with Hugging Face

> **What this guide is.** A guided tour of the assignment: the ideas behind each part, what you are supposed to *notice*, how to approach the code, and where students usually get stuck.
> **What it is not.** It does not contain the answers to the graded questions. You will see small demonstrations on *other* examples than the ones in your notebooks, and many "predict first, then run" prompts. Doing the predicting yourself is the point.

Read it once before you start, then keep it open next to the handout (`README.md`).

---

## Contents

0. The mental model (read this first)
1. How the work is organised
2. Part A — The ecosystem
3. Part B — Pretrained tokenizers
4. Part C — Encoding, decoding, vocabulary
5. Part D — Normalization and pre-tokenization
6. Part E — Training your own tokenizers
7. Part F — Saving, reloading, wrapping, testing
8. Part G — The multilingual benchmark
9. Part H — Report and final challenge
10. Debugging cheat sheet
11. Self-check before submitting

---

## 0. The mental model

### 0.1 A tokenizer is a pipeline, not a vocabulary

In Assignment 01 you wrote BPE: a *model* that learns merges. A real tokenizer wraps that model in several other steps. Memorise this picture; every part of Assignment 02 is about one box of it.

```
          ENCODING (text -> ids)
 raw text
    │
    ▼
┌────────────┐   ┌───────────────┐   ┌──────────────────┐   ┌────────────────┐
│ normalizer │ ► │ pre-tokenizer │ ► │ MODEL            │ ► │ post-processor │ ► ids
│ clean-up   │   │ first split   │   │ BPE / WordPiece  │   │ add [CLS] etc. │
└────────────┘   └───────────────┘   │ / Unigram        │   └────────────────┘
                                     └──────────────────┘
          DECODING (ids -> text)
 ids ► DECODER ► text
```

| Box | Question it answers | Example of what it can do |
|---|---|---|
| normalizer | "Which different strings should be treated as the same?" | lowercase, remove accents, Unicode NFC/NFD |
| pre-tokenizer | "Where may the model *never* merge across?" | split on spaces, on punctuation, into bytes |
| model | "How do I split each piece into known sub-pieces?" | BPE merges, WordPiece greedy matching |
| post-processor | "Which special tokens frame the sequence?" | `[CLS] … [SEP]` |
| decoder | "How do I turn tokens back into readable text?" | glue `##` pieces, map bytes back to characters |

**Two consequences you must understand deeply:**

1. The *vocabulary alone does not define the tokenizer.* Two tokenizers with the same vocabulary but different normalizers behave differently.
2. Normalization can be **lossy** (information destroyed, cannot be undone) while byte-level handling is designed to be **lossless**. Keep that contrast in mind in Parts C, D, E and G.

### 0.2 Two libraries, two layers

| | `tokenizers` | `transformers` |
|---|---|---|
| What | the engine (written in Rust) with the pipeline above | a model-oriented wrapper around it |
| Object you meet | `tokenizers.Tokenizer` | `AutoTokenizer` / `PreTrainedTokenizerFast` |
| You use it for | building, training, saving tokenizers | loading pretrained ones, batching, padding, `save_pretrained` |
| Encode API | `tok.encode(text)` returns an `Encoding` object (`.tokens`, `.ids`, `.offsets`) | `tok.tokenize(text)`, `tok.encode(text)`, `tok(text)` |

A *fast* Transformers tokenizer **contains** a `tokenizers.Tokenizer` (you can reach it as `tok.backend_tokenizer`). That is why you will manipulate the same pipeline in both worlds, and why Part G needs an adapter: the two worlds expose different method names for the same thing.

### 0.3 The five words you will use constantly

* **token** — a string piece the model works with (`"Ġthe"`, `"##ing"`, `"▁hello"`).
* **id** — the integer index of that token in the vocabulary. Arbitrary labels, not measurements.
* **vocabulary** — the table token ↔ id.
* **special token** — a control symbol (`[UNK]`, `[PAD]`, `[CLS]`…) that is not ordinary text.
* **unknown token** — what a tokenizer emits when it has no way to represent a piece. Whether this can happen at all depends on the design: keep an eye on it all through the assignment.

---

## 1. How the work is organised

### 1.1 Test-driven loop

```
read the docstring → implement → pytest (file for this part) → run the notebook → write your analysis
```

Run tests **by file**, not all at once, and read the *first* failure only:

```bash
pytest tests/test_bpe.py -x
pytest tests/test_wordpiece.py -x
pytest tests/test_tokenizer_io.py -x
pytest tests/test_evaluation.py -x
```

How to read a failing test:

1. Look at the **name** — it describes the behaviour (`test_special_tokens_are_atomic`).
2. Look at the **assert line** that failed — it shows expected vs actual.
3. Reproduce it in a notebook cell with the same input and `print` the intermediate values.
4. Only then change the code.

If you see `NotImplementedError`, you have simply not written that function yet. If a test fails because of a *missing library or file*, check your environment (`pip install -r requirements.txt`) and the working directory (run `pytest` from the repo root).

### 1.2 Suggested order

```
notebook 01 (A, B) → 02, 03 (C) → 04, 05 (D)      ← 🧩 exercises + exploration
notebook 06, exercises 6.1–6.6  → src/train_bpe.py, src/train_wordpiece.py   → pytest → rest of notebook 06 (E)
notebook 06, exercises 6.8–6.9  → src/tokenizer_io.py                        → pytest → your own tests (F)
notebook 07, exercises 7.1–7.4  → src/evaluation.py                          → pytest → benchmark (G)
report + final challenge
```

Explore first, build second: what you observe in the pretrained tokenizers (Parts B–D) tells you what to look for when you evaluate your own (Parts E–G).

### 1.3 The 🧩 code-completion exercises — how to use them

The notebooks contain 26 exercises where part of the code is replaced by `____`. They exist because the Hugging Face calls you complete here (`AutoTokenizer.from_pretrained`, `tok.tokenize`, `Tokenizer(BPE(...))`, `BpeTrainer(...)`, `tokenizer.train(...)`, `PreTrainedTokenizerFast(...)`) are the ones you will type again and again in LLM projects. Reading them is not enough; you must write them.

**A method that works:**

1. **Read the exercise statement and the hints first.** The hints name the function or attribute you need and give the documentation page. Decide what *type* of thing each blank must be (a class instance? a string? a list? a method call?).
2. **Look at the surrounding code**, not just the blank. In `tokenizer.decoder = ____` the left-hand side tells you it must be a *decoder object*; the imports at the top of the cell tell you which ones are available.
3. **Look it up before guessing**: the docs, or `help(SomeClass)` / `SomeClass?` in the notebook, show the constructor arguments. Try small things in a scratch cell (`print(type(x))`, `dir(x)`).
4. **Run the cell, then the ✅ check cell.** If the check fails, read the assertion: it tells you what it expected. Fix, re-run.
5. **Explain it to yourself in one sentence** ("this line tells the tokenizer to treat text as bytes before BPE"). If you cannot, you copied; redo it.
6. **Then package it.** Where an exercise corresponds to a function in `src/`, the docstring says which one. Your `src/` function is usually the exercise code with parameters instead of constants.

**Reading an error message** is part of the skill: `NameError: name '____' is not defined` means a blank is still there; `TypeError` about arguments means wrong constructor arguments (check the docs); `AttributeError` means you used a method that object does not have (check `dir(obj)`).

### 1.4 A habit that pays off all semester: *predict, run, explain*

For every experiment in the notebooks, write one sentence **before running the cell**: "I expect X because Y." Then run it. When the output disagrees with you, you have found something worth explaining in your report — that is where the marks are.

---

## 2. Part A — The ecosystem

**Goal:** know which tool does what, so that later you reach for the right one.

### How to approach it

Do not memorise. Build a one-line identity for each library from its own documentation front page, then test yourself with a concrete job:

* "I want to download a tokenizer someone published." → which library? which class?
* "I want to train a new vocabulary on my text." → ?
* "I want to pad a batch of sentences and get attention masks." → ?
* "I want to stream a 100 GB text corpus." → ?
* "I want to upload my tokenizer so a teammate can load it by name." → ?

If you can route those five jobs correctly, you understand the ecosystem.

### Concepts to look up

* `AutoTokenizer`: what does "Auto" refer to? Which file does it read to decide which class to build?
* "Pretrained" tokenizer: pretrained on *what*, and by *whom*? Why does this matter to you as a user (hint: think about the languages in Part G).
* Fast vs slow: where does the speed come from, and what *features* only fast tokenizers have (offsets!)?
* Hub: what files are actually in a model repository, and which of those a tokenizer needs.

### Pitfall

Don't answer "tokenizers is for tokenizing and transformers is for transformers". The assignment wants the *relationship* (one wraps the other) and the *division of labour*.

---

## 3. Part B — Pretrained tokenizers

**Goal:** learn to *interrogate* a tokenizer: what is it, how big, what special tokens, what is inside.

🧩 *Exercises 1.1–1.3:* load the three tokenizers (`AutoTokenizer.from_pretrained`), read their attributes, and reach the wrapped `backend_tokenizer` and its four components. After each exercise, ask what the check cell *does not* verify.

### 3.1 The table you build

For each of GPT-2, BERT, FLAN-T5 collect: class, fast or not, `vocab_size`, `len(tokenizer)`, special tokens. Two things deserve your attention:

* **The two sizes.** One is "the base vocabulary", the other "everything the tokenizer can output". Find out which is which, and which models show a difference. When you later report "vocabulary size" in Part G you will have to choose, and you must say which one you chose and why.
* **The special tokens differ in kind, not just in name.** Compare which models have a `[CLS]`, a `[MASK]`, a padding token, a beginning-of-sequence token. Ask: what job was each model designed for (reading? generating? both?) and how does the special-token set reflect it?

### 3.2 Looking inside

Notebook 01 prints the components of each backend tokenizer (model, normalizer, pre-tokenizer, decoder). This is the same box diagram from §0.1. **Draw it for each of the three models** on paper — a table with five columns and three rows. You will refer to it in Parts C, D and G.

### 3.3 Configuration files

A tokenizer repository contains a few files with names like `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, and, depending on the model, vocabulary/merge files or a SentencePiece model. For each: *what information does it carry that the others don't?* Part F asks you to think about exactly this question for your own tokenizer.

### Questions to think about while exploring

* Why would two research groups choose different tokenizers for similar models?
* What does the vocabulary size trade off? (Think: size of the embedding table vs. length of sequences — Assignment 01 Part F.)
* Why must special tokens be protected from being split by the model algorithm?

---

## 4. Part C — Encoding, decoding, vocabulary

🧩 *Exercises 2.1–2.3 and 3.1–3.3* are the hands-on core of this part: `tokenize` / `convert_tokens_to_ids` / `convert_ids_to_tokens` / `decode`, `encode` with and without special tokens, batching with `padding=True`, and then `get_vocab`, inverting a vocabulary, id and token lookups, finding multi-piece words.

### 4.1 Three levels of conversion

```
text ──tokenize──► tokens (strings) ──convert_tokens_to_ids──► ids
text ──encode───────────────────────────────────────────────► ids      (+ special tokens by default!)
ids  ──decode───────────────────────────────────────────────► text
```

Experiment: on one sentence, run `tokenize`, `encode`, and `encode(..., add_special_tokens=False)` for every tokenizer and **compare lengths**. Predict first: which of the three tokenizers add something by default, and where?

That observation is the origin of a rule you will implement in Part G: *to compare tokenizers fairly you must count tokens under identical conditions.*

### 4.2 The round trip and why it can disagree

Notebook 02 shows `original`, `tokens`, `ids`, `decoded`. You must explain **every** difference between `original` and `decoded`. A good method is to sort differences into causes:

| Cause family | Question to ask | Where to look |
|---|---|---|
| information destroyed on the way in | did the normalizer change the text? | Part D, notebook 04 |
| formatting reconstructed on the way out | did the decoder re-insert/clean spaces? | the decoder box |
| framing tokens | are there `[CLS]`/`</s>` in the decoded string? | `skip_special_tokens` |
| unknowns | did a piece map to the unknown token? | the unknown-token id |

For each mismatch you find, name the family *and* the component responsible. "It's slightly different" is not an explanation.

### 4.3 What happens with something the vocabulary does not contain?

Try inputs that are far from English: another script, an emoji, a rare word. For each tokenizer, answer three things: *is there an unknown token in the output? how many tokens did it take? can I get the original back?* You will see that the three tokenizers have **three different philosophies** here. Write them down in your own words — they become the backbone of Part G's analysis.

### 4.4 Exploring a vocabulary

`get_vocab()` gives a dict token → id. Things to do and think about:

* Print 50 tokens spread over the id range, not the first 50 (the beginning is often special tokens or single characters and tells you little).
* Look up a word, a word with a leading space, a word with a capital letter. Notice that "the word" is not one entry but several candidates — why?
* Find words that need several tokens. Do the pieces look like linguistic units (prefixes, suffixes) or arbitrary chunks? What does that tell you about how the vocabulary was built (frequency, not linguistics)?
* **Symbols you will meet.** `Ġ`, `▁`, `##` are markers of word boundaries/continuations used by different families. Work out from examples which marker means "this token starts a new word" and which means "this token continues the previous one" — the logic is *different* between families, and it is a classic exam question.
* **Ids are labels.** Is id 5 "more important" than id 5000? Does the id order mean anything for each tokenizer (frequency? alphabetical? merge order?). Investigate with data, don't guess: take a few tokens you know are very common and look at their ids.

### Mini-demo (byte-level view, *not* in your notebooks)

```python
from tokenizers.pre_tokenizers import ByteLevel
ByteLevel(add_prefix_space=False).pre_tokenize_str("naïve  café")
# [('naÃ¯ve', (0, 5)), ('Ġ', (5, 6)), ('ĠcafÃ©', (6, 11))]
```

Read it slowly: `ï` became two visible characters because it is **two UTF-8 bytes**; each space became `Ġ`; the *double* space produced a stand-alone `Ġ`. Nothing was dropped — which is why byte-level tokenization can be lossless. Keep this in mind for Part D.

---

## 5. Part D — Normalization and pre-tokenization

### 5.1 Normalization: making different strings equal (on purpose)

🧩 *Exercises 4.1–4.2:* read a pretrained tokenizer's normalizer, then **build** `Sequence([NFD(), StripAccents(), Lowercase()])` and a `BertNormalizer`. Before filling the blanks, predict what happens if you swap the order of two of them.

Run the eight inputs through the three tokenizers (notebook 04). Build a small matrix in your head: for each pair of inputs that differ only by capitalization / accents / punctuation, *which tokenizers return identical tokens for both?* The tokenizers that do are applying a normalizer; the ones that don't are preserving the distinction.

Two ideas that students mix up:

* **Lowercasing / accent stripping** change *letters*.
* **Unicode normalization (NFC/NFD/NFKC/NFKD)** changes the *code points used to represent the same visible character*. Example (`ñ`):

```python
import unicodedata
[hex(ord(c)) for c in "ñ"]                                   # ['0xf1']          one code point
[hex(ord(c)) for c in unicodedata.normalize("NFD", "ñ")]     # ['0x6e', '0x303'] n + combining tilde
```

The second form is exactly why "strip accents" is usually implemented as *decompose, then delete the combining marks*:

```python
from tokenizers.normalizers import Sequence, NFD, StripAccents, Lowercase
Sequence([NFD(), StripAccents(), Lowercase()]).normalize_str("Ñandú")   # 'nandu'
```

**Normalizers compose** (`Sequence`). The notebook asks you to build and compare several; look at what each does to Arabic text with diacritics and to compatibility characters. Ask yourself: *would a language where accents distinguish different words suffer from "strip accents"?* That is the heart of the question "why must normalization depend on the tokenizer's training?".

### 5.2 Pre-tokenization: where merges are forbidden to cross

🧩 *Exercises 5.1–5.2:* instantiate the four pre-tokenizers, run them with `pre_tokenize_str`, and chain three of them with `Sequence`.

A pre-tokenizer splits text into *pre-tokens*; the model then works **inside** each pre-token and can never merge across a boundary. Compare on a string that is *not* in your notebook:

```python
text = "x-ray 3.5kg, ok?"
Whitespace().pre_tokenize_str(text)
# x | - | ray | 3 | . | 5kg | , | ok | ?        <- splits on punctuation, drops spaces
WhitespaceSplit().pre_tokenize_str(text)
# x-ray | 3.5kg, | ok?                          <- splits on spaces only
BertPreTokenizer().pre_tokenize_str(text)
# x | - | ray | 3 | . | 5kg | , | ok | ?        <- punctuation split, like Whitespace here
```

Each item is `(piece, (start, end))`: the **offsets** point back into the original string. Questions to ask of every output:

* Where did the punctuation go? Attached to a word, or separated?
* What happened to spaces — kept, attached to the next piece, or dropped? (Compare with the byte-level demo above.)
* Is the original text recoverable from the pre-tokens alone? What would have to be recorded?
* What **cannot** now be learned as a single token? (Look at `x-ray` and `3.5kg` under each pre-tokenizer.)

Then answer the assignment's real question: *why does the choice of pre-tokenizer change the final token sequence even if the model algorithm is identical?* Use a concrete example from your own run.

Notice the case of "Hello   world" (three spaces) in the assignment: which pre-tokenizers make it indistinguishable from "Hello world"? What does that mean for a tokenizer that claims to be *lossless*?

---

## 6. Part E — Training your own tokenizers

This is the part with the most code. Slow down and understand the *design*, then the code is short.

### 6.1 What "training a tokenizer" means

You give a trainer text; it learns the **model's** table (merges for BPE, subword pieces for WordPiece). Everything else — normalizer, pre-tokenizer, decoder — you **choose**; they are not learned. So a trainable tokenizer has two kinds of parts:

| Chosen by you (design) | Learned from data |
|---|---|
| normalizer, pre-tokenizer, decoder, special tokens, vocabulary size, `min_frequency` | the vocabulary and merge rules |

### 6.2 Byte-level BPE — design walk-through

🧩 **You write this in notebook 06, exercises 6.1–6.3** (pipeline → trainer → train and inspect), *then* package it in `src/train_bpe.py`.

1. **`build_bpe_tokenizer`** assembles the *untrained* pipeline: BPE model + ByteLevel pre-tokenizer + ByteLevel decoder. Check each box against the diagram in §0.1. Before coding, answer: *what would decoding return if I forgot the decoder?* (Try it on an accented word and look at the output; you saw the cause in the mini-demo above.)
2. **`make_trainer`** builds a `BpeTrainer`. Three parameters to understand:
   * `vocab_size` — a **target**, not a promise. Part of notebook 06 maps when the final size differs.
   * `min_frequency` — a pair must occur at least this often to be merged. Predict how this interacts with a *small* corpus versus a *large* one.
   * `initial_alphabet=ByteLevel.alphabet()` — gives the trainer all 256 byte symbols. Ask: *what does a tokenizer built on all 256 bytes never need to say?* And what would be missing if a byte never occurred in your corpus and you hadn't included it?
3. **`train_bpe`** wires them together. The stub tells you the traps: normalize the "files" argument to a list of strings, put the unknown token first, return the trained tokenizer. Remember special-token **order = ids** (the first special token gets id 0).
4. **`train_bpe_from_texts`** does the same from an in-memory iterator. A test checks that both routes give the same vocabulary — a good consistency idea you can reuse.

After training, *look* before you evaluate:

```python
enc = tok.encode("the cat")
enc.tokens, enc.ids, enc.offsets
# ['t', 'he', 'Ġc', 'at'], [87, 266, 277, 268], [(0,1),(1,3),(3,5),(5,7)]   (from a tiny toy run)
```

`Ġc` is "a space, then `c`". Note how the leading space is *part of the token*: this is how byte-level BPE remembers spacing without a separate space token.

### 6.3 WordPiece — design walk-through

🧩 **Notebook 06, exercises 6.4–6.6** (pipeline → trainer + `train_from_iterator` → `TemplateProcessing` post-processor that adds `[CLS]`/`[SEP]`), then `src/train_wordpiece.py`. This pipeline has **four** configured boxes instead of two:

* `BertNormalizer(lowercase=True, strip_accents=True)` — lossy by design.
* `BertPreTokenizer()` — splits on spaces *and* punctuation.
* WordPiece model with `[UNK]` and the `##` continuation prefix.
* The WordPiece decoder, which glues `##` pieces back together.

WordPiece segments a (normalized, pre-tokenized) word by repeatedly taking the **longest vocabulary piece** that matches from the left; later pieces carry `##`. If *no* piece matches at some position, the whole word becomes the unknown token. Hold that last sentence in your mind: it is a different failure mode from the byte-level design, and Part G will let you measure the difference.

### 6.4 The experiments in notebook 06 — what each is for

| Experiment | What you should be able to say afterwards |
|---|---|
| 20 sentences with both tokenizers | which differences come from normalization, which from the algorithm, which from unseen characters |
| requested vs final vocabulary size over a grid | *when* the final size is smaller, and the corpus property that causes it |
| effect of `min_frequency` | its role on a small vs a large corpus |
| **reproducibility experiment**: train the same tokenizer several times and compare vocabularies | whether "training is deterministic" is true for each algorithm, and what that implies for how you *publish* a tokenizer |

For the reproducibility experiment, decide **in advance** how you will compare two vocabularies (equal dictionaries? same set of tokens? overlap fraction?). A good habit: choose a measure that tells you *how different*, not only *whether* different.

### 6.5 Common Part E mistakes

* Passing `Path` objects where the trainer wants strings.
* Forgetting that `train` mutates the tokenizer (you train an object; you don't get a new one back from the trainer).
* Comparing a 2000-token tokenizer trained with different settings and concluding something about the *algorithm*. Change one thing at a time.
* Forgetting that the unknown token must also be a special token in the trainer list.
* Ignoring special token ids: tests check them because downstream code (padding!) relies on them.

---

## 7. Part F — Saving, reloading, wrapping, testing

### 7.1 Why saving is a first-class concern

After training you hold an object in memory. To reuse it tomorrow, or to give it to someone else, you must serialise *everything that defines its behaviour*. Open the JSON of your saved tokenizer in notebook 06 and list the top-level keys; for each key ask: *if this were missing, how would behaviour change?* That list is your answer to "why is saving only the vocabulary insufficient".

### 7.2 The functions

🧩 **Notebook 06, exercises 6.8–6.9** give you the raw calls (`save`, `Tokenizer.from_file`, `PreTrainedTokenizerFast(tokenizer_file=…)`, `save_pretrained`, `AutoTokenizer.from_pretrained`). The functions below package them.

`src/tokenizer_io.py`:

* `save_tokenizer` / `load_tokenizer` — thin wrappers, but make parent folders and return the path.
* `find_mismatches(before, after, sentences)` — your **consistency checker**. Design question: what counts as "the same"? (ids? tokens? decoded text? all three?) The tests tell you what they expect; think about why a check on ids alone might miss something.
* `to_transformers` — wraps your `tokenizer.json` in a Transformers fast tokenizer. Key idea: the raw tokenizer knows *how to split*, but not *which of its tokens is the padding token*. The wrapper is where you declare that. Declare only what exists in your tokenizer: your BPE has `[BOS]`/`[EOS]` and no `[CLS]`; your WordPiece has `[CLS]`/`[SEP]` and no `[BOS]`.

### 7.3 What the wrapper buys you

Try, after wrapping: batching two sentences of different lengths with `padding=True`; inspect `attention_mask`; call `save_pretrained(folder)` and reload with `AutoTokenizer.from_pretrained(folder)`. Predict the mask for the shorter sentence **before** running it. Then think about why neural networks need that mask — this is the bridge to everything you will do next semester.

### 7.4 Writing your own tests (graded)

The assignment requires automated tests for *your* saved tokenizers. Put at least five in `tests/test_my_tokenizers.py`, one per behaviour. A good test is small, named after the behaviour, and has one clear assertion. Cover:

1. the saved file exists and loads,
2. a known sentence encodes to non-empty ids,
3. `decode(ids)` returns what you expect for your tokenizer (careful: expected is not always "identical to input" — that depends on lossy vs lossless!),
4. each special token has the id you declared,
5. encoding is identical after save → reload.

Think about *negative* tests too: something your tokenizer must **not** do (for example, produce an unknown token on some input).

---

## 8. Part G — The multilingual benchmark

This is worth the most points and is where good reports separate themselves. Think of it as designing a **fair experiment**.

### 8.1 What are we comparing, on what?

Five tokenizers: GPT-2, BERT, FLAN-T5 (pretrained) and your BPE and WordPiece. One shared evaluation set of 300 sentences: English, French, Modern Standard Arabic (MSA), Algerian Darija. Look at `data/README.md` first: what the sentences are, who wrote them, what the training corpus contains **and does not contain**. A benchmark can only be interpreted if you know the relationship between the training data and the test data.

### 8.2 Fairness rules you will implement

1. **Same sentences for every tokenizer.**
2. **No special tokens counted.** If one tokenizer adds two framing tokens and another none, a "tokens per sentence" comparison is biased by exactly that difference. (You noticed it in §4.1.)
3. **Same definition of the unit.** What is a "character"? (`len(str)` counts Unicode code points.) What is a "word"? (whitespace-separated — which is not the same thing in Arabic, French and English; be prepared to discuss it.)
4. **Metrics from totals, not from averages of ratios.** If sentence A has 10 characters and 5 tokens, and sentence B has 100 characters and 10 tokens, is the overall "characters per token" `(2 + 10) / 2` or `110 / 15`? Work out which one answers "how many characters does one token cover *on this dataset*", and why the other overweights short sentences. The tests contain a tiny example designed to catch this.
5. **Train and evaluation data separated.**

### 8.3 The adapter (`TokenizerAdapter`)

🧩 *Exercise 7.1* is the adapter in miniature: count the tokens of one sentence through each API. *Exercise 7.2* computes the metrics by hand with pandas, and the check compares your numbers with your `summarize()`: if they disagree, one of the two is wrong — find out which.

The problem: one object is a `tokenizers.Tokenizer`, the other a Transformers tokenizer, and the method names differ (§0.2). The adapter gives both the same four things: `tokenize`, `encode`, `vocab_size`, `unk_id`. The stub tells you which call to use for each family. Two details deserve thought:

* **Which vocabulary size?** For Transformers tokenizers there are two candidates (you met them in Part B). The stub names the one that counts *everything*. Make sure the number in your table means the same thing for all five rows.
* **How do you find the unknown token?** A raw tokenizer stores it in its *model*; a Transformers tokenizer exposes it as an attribute. A tokenizer with *no* unknown token at all should not crash your code.

### 8.4 The metrics, in plain language

| Metric | Reads as | Higher is… |
|---|---|---|
| average tokens per sentence | "how long are my sequences?" | worse (longer = costlier) |
| maximum tokens | "what is my worst case?" (matters for context windows) | worse |
| characters per token | "how much text does a token cover?" | better (more compact) |
| bytes per token | same idea in UTF-8 bytes — lets you separate "this script is expensive in bytes" from "this tokenizer is inefficient" | better |
| tokens per word | "fertility": how many pieces per word | worse |
| **unknown-token rate** | "how often did the tokenizer give up?" | worse — and it **changes how you must read all other columns** |
| encoding time | speed | lower is better, but see below |
| parity | cost in language L ÷ cost in the reference language | closer to 1 = fairer |

**Read `unk_rate` first, always.** A tokenizer that gives up on many words can produce *short* sequences. Ask of every row: *is this short because the tokenizer is efficient, or because it discarded information?*

### 8.5 Timing properly

Single measurements are noise. The provided `time_encoding` does a warm-up pass, then several timed passes and averages. In your report say: which machine, how many repeats, whether you encoded one sentence at a time or in batches (and why that matters for a Rust-backed library), and what you will *not* claim from the numbers.

### 8.6 Building the tables and figures

Required outputs and what each should let a reader *see*:

| Output | A reader should be able to answer… |
|---|---|
| sentence-level CSV | "show me the raw data behind any number" |
| overall comparison table | "who is shortest / most compact / fastest / has the most unknowns overall?" |
| table per language | "does the ranking change with the language?" |
| bar chart: average tokens per language | "how does cost vary by language for each tokenizer?" |
| sequence-length distributions | "is the difference in the mean caused by a few long sentences or by everything?" |
| characters per token | "how compact is each tokenizer per language?" |
| vocabulary sizes | "does a bigger vocabulary explain the pattern?" (careful!) |

Figure hygiene: label axes with units, put the language or tokenizer in a legend, and make the **same tokenizer the same colour/order** in all figures.

### 8.7 How to interpret — a method

For each analysis question in notebook 07, answer in four steps: **claim → number → mechanism → caveat.**

* *Claim*: "Tokenizer X is the most compact on French."
* *Number*: the exact cell(s) of your table.
* *Mechanism*: why — what in the pipeline (training data? vocabulary size? normalization? byte-level fallback?) produces it.
* *Caveat*: what the number does **not** show (small evaluation set, one author, whitespace word as unit, one training domain…).

Use your Part B–D knowledge to explain mechanisms. For instance, to explain why a tokenizer is expensive on a script, ask *which of the three philosophies from §4.3 it follows* and what that philosophy costs when the script is absent from training.

### 8.8 Questions to keep asking while you read your results

* Is the tokenizer with the fewest tokens *the best*? What else would you need to know (unknown rate, information preserved, downstream accuracy, speed, memory)?
* Does a bigger vocabulary necessarily give shorter sequences? Find two tokenizers in your own table that let you answer with evidence.
* Is Darija more or less costly than MSA for each tokenizer? Propose **two different** possible explanations and say what extra data would let you decide between them.
* Why does sequence length matter downstream? (Recall what self-attention does with sequence length; you can answer in terms of cost growth without any neural-network code.)
* If parity with English is far from 1 for a language, what are the consequences for a *user* of that language (cost, context window, quality)?

---

## 9. Part H — Report and final challenge

### 9.1 The report as an argument

Don't narrate what you did in order; argue a position supported by your measurements. A useful backbone:

1. *Question*: how do tokenizer design and training data shape efficiency across languages?
2. *Method*: the pipeline view (§0.1), your data, your metrics, your fairness rules.
3. *Findings*: three or four statements, each backed by a table or figure.
4. *Explanation*: mechanisms (normalization, pre-tokenization, model, decoder, training data).
5. *Limits*: be specific. "Small dataset" is weak; "50 Darija sentences by a single author, whitespace words as the unit" is strong.

Use `report/REPORT_TEMPLATE.md`. Every number must be traceable to a file in `results/`; include library versions and commands in the appendix.

### 9.2 The final challenge — how to think about it

You are designing, not training. Treat it as a short research proposal on one question: *with a 32 000-token budget, how do I make a tokenizer that serves four languages reasonably fairly?* Structure your 1–2 pages around the five prompts of the handout, and reuse **your own measurements** as evidence:

* **Algorithm choice** — connect it to what you saw about unseen scripts under different designs.
* **Data allocation** — vocabulary share follows training-data share. Compare simple strategies (proportional to size, equal per language, something in between) and say what each would do to a language with much less text.
* **Keeping any one language from dominating** — which knobs are available *before* training (data) and *during/after* (vocabulary allocation, evaluation-driven iteration)?
* **Fairness measurement** — pick a metric, a reference language, and justify. Note that the choice of reference language changes the numbers.
* **Vocabulary size vs sequence length** — use the trend in your own results; mention the cost of a large vocabulary on the model side (you computed it in Assignment 01).

Propose **an experiment you could actually run**: what you vary, what you measure, what result would make you change your mind.

---

## 10. Debugging cheat sheet

| Symptom | Likely cause | What to check |
|---|---|---|
| `NotImplementedError` | function not written yet | the stub named in the traceback |
| Decoded text is full of `Ã©`, `Ø§` | no / wrong decoder on a byte-level tokenizer | `tok.decoder` |
| WordPiece decode shows `play ##ing` | WordPiece decoder missing | decoder box |
| Special token ids are not 0, 1, 2… | order of the `special_tokens` list | trainer arguments |
| `ValueError`/type error in `train` | `Path` objects passed instead of strings | convert with `str()` |
| Final vocabulary smaller than requested | corpus ran out of pairs above `min_frequency` | corpus size, `min_frequency` |
| `PreTrainedTokenizerFast` complains about the file | wrong path / you passed a folder | pass the path of `tokenizer.json` |
| `pad_token` is `None` after wrapping | not declared | arguments of `to_transformers` |
| Benchmark counts differ from tests' tiny example | special tokens counted, or mean-of-ratios | adapter, `summarize` |
| `KeyError` in `parity` | language label spelled differently | the `language` column values |
| Numbers differ from a classmate's WordPiece | WordPiece training is not perfectly reproducible | notebook 06 experiment — compare using the *saved* files |
| Pretrained download fails | no internet / Hub unreachable | try again online; tests for pretrained models are skipped, but notebooks 01–04 need them |
| Notebook edits to `src/` not visible | stale import | restart kernel (the notebooks use autoreload, but restarting is safest) |

---

## 11. Self-check before submitting

**Code**
- [ ] `pytest` passes everywhere except pretrained tests only if you were offline (and you ran them online at least once).
- [ ] `tests/test_my_tokenizers.py` has at least five tests and they pass.
- [ ] You did not edit the provided tests.
- [ ] All 26 🧩 exercises are completed (no `____` left) and every ✅ check cell passes.

**Experiments**
- [ ] Notebooks 01–07 re-run top to bottom without errors.
- [ ] `models/bpe/tokenizer.json` and `models/wordpiece/tokenizer.json` are committed and are the ones you evaluated.
- [ ] `results/` contains the sentence-level CSV, the comparison tables and four figures.

**Understanding** — can you explain, without notes:
- [ ] the five boxes of the pipeline and what each does to *one concrete example*;
- [ ] why saving the vocabulary alone is not enough;
- [ ] why a short sequence is not automatically a good tokenization;
- [ ] the difference between "the tokenizer failed and said so" and "the tokenizer succeeded at higher cost";
- [ ] one thing in your benchmark that could be wrong, and how you would test it?

**Report**
- [ ] Every number comes from your files in `results/`.
- [ ] Data sources and language distribution documented.
- [ ] Final challenge included (1–2 pages) and tied to your results.
- [ ] References cited; any AI assistant or external help acknowledged.

---

*Good luck — and remember the habit that matters most: predict, run, explain.*
