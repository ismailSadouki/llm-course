# Assignment 02 — Beginner's Walkthrough
### Text Processing and Tokenization with Hugging Face — step by step, for people who are new to programming

> **Who this is for.** You are new to Python, or you know a little but the Hugging Face libraries look like a wall of unfamiliar names. This guide goes slowly: it explains the Python you need, how to *explore* an unfamiliar library, the ideas behind every part, and how to think when you are stuck.
> **What it does not do.** It does not give you the answers to the graded exercises or questions. It gives **worked examples on different problems** (a toy "cat/dog" tokenizer, small tables of numbers…) so that you learn the technique and then apply it to your own task.
>
> If you are comfortable with Python, read `WALKTHROUGH.md` instead (shorter, more conceptual). If you have not done Assignment 01, read its beginner guide first: this one assumes you know what a token, a vocabulary and an id are (they are recalled in §4).

> **A note about the examples.** The Hugging Face Hub (where GPT-2, BERT and FLAN-T5 live) needs internet. The demonstrations in this guide use a **tiny tokenizer trained on three sentences**, so that every example runs anywhere. Your numbers with the real models will be different — that is the point of the exercises.

---

## Contents

0. Read this first
1. Setting up
2. A Python survival kit (new things for this assignment)
3. How to explore a library you do not know
4. The ideas, in plain words
5. How to work: exercises, errors, tests
6. Notebook 01 — the ecosystem and pretrained tokenizers (Parts A, B)
7. Notebooks 02 and 03 — encoding, decoding, vocabularies (Part C)
8. Notebooks 04 and 05 — normalization and pre-tokenization (Part D)
9. Notebook 06 — training your own tokenizers (Parts E, F)
10. Packaging the code in `src/`
11. Writing your own tests
12. Notebook 07 — the multilingual benchmark (Part G)
13. The report and the final challenge
14. Troubleshooting FAQ
15. Glossary
16. Plan and final checklist

---

## 0. Read this first

### 0.1 It is normal to feel lost

The libraries used here are the daily tools of people who build language models. They have *many* classes and functions. Nobody memorises them: professionals **look things up all the time**. The skill you are learning is how to *find* the right tool, *try* it on a small example, and *check* the result.

Three rules that make everything easier:

1. **Work in tiny steps.** Write 2–3 lines, run, look, continue.
2. **Print everything.** If you don't know what something is, `print(type(x), x)`.
3. **Predict, then run.** Before running a cell, say: "I expect …". When the result differs, you learned something — that is what the report is made of.

### 0.2 The two kinds of work

| | What it is | Where |
|---|---|---|
| 🧩 **Exercises** (26) | a few lines of Hugging Face code with `____` blanks; hints name the functions to use; a ✅ check cell tells you if you are right | the notebooks |
| 🛠 **Functions** | you package the exercise code into proper functions that automatic tests can check | `src/*.py` |

You do the exercise first (small, with instant feedback), then move it into `src/` (tested).

### 0.3 How long will it take?

Plan **14–18 hours** if you are a beginner (the handout says 8–12 for experienced students). Spread it over two weeks. See §16.

### 0.4 How to ask for help

Say **what you tried**, **what you expected**, **what happened** (copy the exact error), and **your code**. "It doesn't work" cannot be answered; "`AttributeError: 'NoneType' object has no attribute 'normalize_str'` on this line of this cell" can.

---

## 1. Setting up

If you did Assignment 01, you know most of this. A refresher:

```bash
cd assignment-02
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import tokenizers, transformers, datasets; print(tokenizers.__version__, transformers.__version__, datasets.__version__)"
pytest                               # red where code is missing: that is the normal starting point
jupyter notebook
```

Things that are new in this assignment:

* **You need internet** the first time you load a pretrained tokenizer. The files are downloaded from the Hugging Face Hub and **cached** on your computer; later runs are offline.
* **The installation is bigger** (several hundred MB). Be patient with `pip install`.
* **Versions differ.** The `transformers` library changes between major versions. If something in the instructions does not match what you see (e.g. the *name* of the class printed for a tokenizer), check the version printed above, read the message carefully, and mention it in your report.
* If a download fails: check your connection, try again, and if it keeps failing tell your teacher (the Hub may be blocked on some networks).

**Notebook reminders:** run cells with **Shift+Enter**, top to bottom; **Kernel → Restart & Run All** gives a clean start; `Esc` then `B` adds a scratch cell below.

---

## 2. A Python survival kit (new things for this assignment)

If you know Python basics (lists, dicts, loops, functions, comprehensions), skim this section for what is new. If not, read Assignment 01's beginner guide §2 first.

### 2.1 Objects have *attributes* and *methods*

An **object** is a thing that carries data and can do things. You write `object.name`:

* `object.name` **without** brackets → an *attribute* (a piece of data);
* `object.name(...)` **with** brackets → a *method* (it does something).

```python
fast.is_fast          # attribute: True
len(fast)             # a function applied to the object
fast.tokenize("hi")   # method call: returns a list
```
The most common mistake is forgetting the brackets (you get something like `<bound method ...>` instead of the result) or adding them when there are none (`TypeError: 'bool' object is not callable`).

### 2.2 Importing from libraries

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
import pandas as pd
```
`from package.submodule import Name` brings `Name` into your notebook. When two things have the same name, rename on import with `as`:

```python
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder    # a different class with the same name
```
You will see exactly this in the exercises.

### 2.3 Keyword arguments

Functions take arguments by position or by **name**:

```python
Trainer(vocab_size=2000, min_frequency=2)       # named: order does not matter
```
Names make code readable and are what the hints refer to (`vocab_size=`, `special_tokens=`…).

### 2.4 Dictionaries and lists you will meet constantly

```python
vocab = {"the": 2, "cat": 3}        # token -> id
vocab.get("dog")                    # None   (no KeyError)
vocab.get("cat")                    # 3
list(vocab.items())[:1]             # [('the', 2)]   first pair
inverse = {i: t for t, i in vocab.items()}      # flip keys and values -> {2: 'the', 3: 'cat'}
sorted(vocab.items(), key=lambda kv: kv[1])     # sort the pairs by their id (kv[1])
```
A **list comprehension with a condition** keeps only some items:

```python
words = ["cat", "tokenization", "dog"]
long_words = [w for w in words if len(w) > 5]       # ['tokenization']
```

### 2.5 Files, folders and JSON

```python
from pathlib import Path
path = Path("models") / "bpe" / "tokenizer.json"     # build a path with /
path.parent.mkdir(parents=True, exist_ok=True)       # create the folder if needed
text = path.read_text(encoding="utf-8")              # read a whole text file
lines = text.splitlines()                            # list of lines
```
**JSON** is a text format for nested dictionaries and lists. Python reads it with `json.loads(text)`, and you get a normal `dict`:

```python
import json
data = json.loads('{"a": 1, "b": [1, 2]}')
print(list(data.keys()))        # ['a', 'b']
```

### 2.6 pandas — tables (you will use it in Part G)

A **DataFrame** is a table with named columns.

```python
import pandas as pd
df = pd.DataFrame({
    "lang":   ["en", "en", "fr", "fr", "fr"],
    "tokens": [10, 12, 20, 22, 30],
    "chars":  [40, 50, 60, 66, 90],
})
df["tokens"]                       # one column (a "Series")
df["tokens"].mean()                # 18.8
df["tokens"].sum()                 # 94
df["chars"].sum() / df["tokens"].sum()     # ratio of totals = 3.255...
df[df["lang"] == "fr"]             # keep rows where the condition is True
df.groupby("lang")["tokens"].mean()          # one value per language
```
`groupby` splits the table by the values of a column and lets you compute something **per group**:

```python
df.groupby("lang")["tokens"].mean().to_dict()      # {'en': 11.0, 'fr': 24.0}
```
`pivot` reshapes a "long" table (one row per tokenizer × language) into a "wide" one (rows = tokenizers, columns = languages):

```python
long = pd.DataFrame({"tok": ["A","A","B","B"], "lang": ["en","fr","en","fr"], "v": [1,2,3,4]})
long.pivot(index="tok", columns="lang", values="v")
# lang  en  fr
# tok
# A      1   2
# B      3   4
```
Note the difference between **an average of ratios** and a **ratio of totals**: for one sentence of 10 chars/5 tokens and one of 100 chars/10 tokens, the first is (2 + 10)/2 = 6, the second is 110/15 = 7.33. They answer different questions (the second is "how many characters does one token cover on this dataset" and it is the one the assignment asks for).

### 2.7 Charts with matplotlib

```python
import matplotlib.pyplot as plt
plt.bar(["en", "fr"], [11.0, 24.0])
plt.ylabel("average tokens")
plt.title("Tokens per sentence")
plt.show()
```
Always give a chart a **title**, **axis labels** and, if there is more than one series, a **legend**. A table can be plotted directly: `table.plot(kind="bar")`.

### 2.8 Measuring time

```python
import time
t0 = time.perf_counter()
sum(range(1_000_000))              # something to measure
elapsed = time.perf_counter() - t0 # seconds
```
One measurement is noisy. The assignment asks for a **warm-up** run (not timed) followed by **several** timed runs, averaged.

### 2.9 Classes and `self` (to read the adapter)

A class bundles data and methods; `self` means "this object". If you have not seen this before, read Assignment 01 beginner guide §2.13. In `TokenizerAdapter`, the object remembers `self.name` and `self.tokenizer`, and its methods use them.

A **property** is a method you call without brackets:

```python
class Box:
    def __init__(self, items): self.items = items
    @property
    def size(self): return len(self.items)

Box([1, 2, 3]).size        # 3   (no brackets!)
```

### 2.10 Bytes (new, and important for byte-level tokenizers)

Text is stored as bytes. One character may take several:

```python
"é".encode("utf-8")            # b'\xc3\xa9'     two bytes
list("é".encode("utf-8"))      # [195, 169]
len("é"), len("é".encode("utf-8"))    # (1, 2)
len("ب"), len("ب".encode("utf-8"))    # (1, 2)   an Arabic letter also takes 2 bytes
```
Keep this in your head: a tokenizer that works **on bytes** sees *two* symbols for many non-English characters. That one fact explains a large part of the benchmark.

---

## 3. How to explore a library you do not know

This is the most useful skill of the assignment. When a hint says "use `BpeTrainer`", you do not need to know it in advance; you need to know **how to find out**.

### 3.1 Four tools

1. **`type(x)`** — what kind of object is this?
2. **`dir(x)`** — what attributes and methods does it have? (ignore the ones starting with `_`)
   ```python
   [name for name in dir(tok) if not name.startswith("_")]
   ```
3. **`help(x)`** or `x?` in Jupyter — the documentation of a function or class: its arguments and what they mean.
4. **The online documentation** (links in the handout): https://huggingface.co/docs/tokenizers/ and https://huggingface.co/docs/transformers/main_classes/tokenizer.

### 3.2 A worked exploration (on a toy tokenizer)

Suppose I just built this tiny tokenizer (don't worry about the details yet — §9 explains them):

```python
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

tok = Tokenizer(WordLevel(unk_token="[UNK]"))
tok.pre_tokenizer = Whitespace()
trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"], show_progress=False)
tok.train_from_iterator(["the cat sat", "the dog sat", "the cat ran home"], trainer=trainer)
```

What can it do? I ask:

```python
type(tok)                                    # <class 'tokenizers.Tokenizer'>
[n for n in dir(tok) if not n.startswith("_")][:12]    # a list that includes 'encode', 'decode', 'get_vocab', 'save', ...
enc = tok.encode("the cat flew")
type(enc)                                    # tokenizers.Encoding
enc.tokens, enc.ids, enc.offsets
# (['the', 'cat', '[UNK]'], [2, 3, 0], [(0, 3), (4, 7), (8, 12)])
```
`flew` was never seen, so it became `[UNK]` (id 0). `offsets` say where each token sits in the original string (`(8, 12)` = characters 8 to 12).

More lookups:

```python
tok.get_vocab_size()            # 8
tok.token_to_id("[PAD]")        # 1
tok.get_vocab()                 # {'[UNK]': 0, '[PAD]': 1, 'the': 2, 'cat': 3, ...}
tok.decode(enc.ids)             # 'the cat'    <- the unknown word disappeared!
```
Notice the last line: **decoding lost information** (the unknown word `flew`). Remember it for Part C.

### 3.3 When an object might be `None`

Some components do not exist: a tokenizer may have **no normalizer**. Then `tok.normalizer` is `None`, and calling a method on it gives `AttributeError: 'NoneType' object has no attribute ...`. Guard against it:

```python
norm = tok.normalizer
if norm is not None:
    print(norm.normalize_str("Hello"))
else:
    print("no normalizer")
```
Writing `type(norm).__name__ if norm else None` is a short way to print "the class name, or None".

---

## 4. The ideas, in plain words

### 4.1 Tokens, ids, vocabulary (recalled from Assignment 01)

```
"I love cats"  →  tokens ["I", "love", "cats"]  →  ids [7, 52, 311]
```
A **vocabulary** is the dictionary token ↔ id. A **special token** (`[PAD]`, `[UNK]`, `[CLS]`…) is a control symbol that is not part of the text.

### 4.2 A tokenizer is an assembly line

Real tokenizers do not only "cut text". They are a **pipeline of stations**; each does one job:

```
raw text
   │
   ▼
[1 normalizer]      clean-up: lowercase? remove accents? standard Unicode form?
   │
   ▼
[2 pre-tokenizer]   first rough cut: on spaces? on punctuation? into bytes?
   │
   ▼
[3 model]           the main algorithm: BPE / WordPiece / Unigram cuts each piece into subwords
   │
   ▼
[4 post-processor]  add the framing tokens: [CLS] … [SEP]
   │
   ▼
ids

ids ──► [5 decoder] ──► text again (glue pieces back together)
```

An analogy: think of an **airport**. Check-in (normalizer) standardises your documents, security (pre-tokenizer) sorts people into queues, the plane (model) carries each group, boarding passes (post-processor) are added at the end, and at the arrival hall (decoder) everybody is put back together.

Why this matters: **two tokenizers can share a vocabulary but behave differently** if their stations differ. And when text comes out "wrong", you ask *which station is responsible?*

### 4.3 Three designs for what happens with the unknown

| Design | Example | What it does with a piece it has no entry for | Cost |
|---|---|---|---|
| "give up" | WordPiece / `[UNK]` | emits the unknown token | information is lost, but sequences are short |
| "fall back to bytes" | byte-level BPE (GPT-2) | writes the piece as bytes, which always exist | never loses information, but unseen scripts cost many tokens |
| "use a subword model" | Unigram / SentencePiece | splits into smaller known pieces | depends on coverage |

These three philosophies are the backbone of Part G. Keep them in mind when you read the benchmark.

### 4.4 Why languages matter

A tokenizer is **trained on text**. Words and scripts that were common in that text get short, efficient tokens; others get cut into many small pieces — or into `[UNK]`. So the same sentence, in two languages, can cost very different numbers of tokens. Since compute and memory grow with the number of tokens, that is a fairness question: is one language more *expensive* to process than another?

### 4.5 Normalization versus Unicode normalization

* **Lowercasing / removing accents** change letters (`Café` → `cafe`).
* **Unicode normalization** keeps the letters but picks one way to write them in memory (`e` + `´` → `é`).
Different jobs; tokenizers often do both.

---

## 5. How to work: exercises, errors, tests

### 5.1 The method for a 🧩 exercise

1. **Read the statement and the hints.** The hints name the classes, functions or attributes.
2. **Say in plain words what the code must do** ("build a tokenizer whose model is BPE").
3. **Explore with the tools of §3** on a scratch cell: `help(BpeTrainer)`, `dir(tok)`.
4. **Find the pattern in a worked example** (this guide shows one for every kind of exercise).
5. **Fill the blank, run the cell, run the ✅ check cell.**
6. **If the check fails:** read the assertion — it says what it expected. Print your variables.
7. **Explain your line in one sentence.** If you cannot, you copied; redo it.

### 5.2 Reading error messages (read the **last line** first)

| Last line says | Usually means | What to do |
|---|---|---|
| `NameError: name '____' is not defined` | a blank is still empty | fill it |
| `NameError: name 'xyz' is not defined` | you skipped a cell, or misspelled a name, or forgot an import | run the cells above; check spelling |
| `ImportError` / `ModuleNotFoundError` | wrong module path or package not installed | check the import line in the exercise; `pip install -r requirements.txt` |
| `OSError: ... couldn't connect to 'https://huggingface.co'` | no internet / Hub blocked | check your connection; retry later |
| `OSError: ... is not a local folder and is not a valid model identifier` | the model name is misspelled | check `"bert-base-uncased"` etc. letter by letter |
| `TypeError: __init__() got an unexpected keyword argument 'xyz'` | you used an argument name that does not exist | `help(TheClass)` shows the real names |
| `TypeError: 'NoneType' object is not callable` / `AttributeError: 'NoneType' ...` | the thing you are using is `None` (e.g. no normalizer) | print it; guard with `if x is not None` |
| `AttributeError: 'X' object has no attribute 'y'` | wrong attribute/method name | `dir(obj)` lists what exists |
| `KeyError: 'abc'` | dictionary has no such key | `d.get(key)` or print the dict |
| `TypeError: ... argument must be str, not PosixPath` | a `Path` was given where a string is needed | wrap it: `str(path)` |
| `AssertionError` in a ✅ cell | your code ran but gave a wrong result | compare with what the check expects |
| `NotImplementedError` | you call a `src/` function that is not written yet | write it |

### 5.3 Reading a failing test

```bash
pytest tests/test_bpe.py -x
```
`-x` stops at the first failure. Read the **test name**, then the **assert line** (expected vs actual), then reproduce that case in a notebook cell and print intermediate values. Fix one failure, re-run **the same file**, repeat.

Order of difficulty: `test_data.py` (already green) → `test_bpe.py` → `test_wordpiece.py` → `test_tokenizer_io.py` → `test_evaluation.py`.

### 5.4 Commit often

```bash
git add .
git commit -m "notebook 06: exercises 6.1-6.3 done"
```

---

## 6. Notebook 01 — the ecosystem and pretrained tokenizers (Parts A and B)

### Part A: the libraries (written answers)

You answer seven questions in your own words. A method that works for each library: open its documentation front page and write **one sentence "this library is for …"**, then test it with a job:

* "I want to download a tokenizer someone published." → ?
* "I want to train a new vocabulary on my text." → ?
* "I want to pad a batch of sentences and get attention masks." → ?
* "I want to load a big text dataset without running out of memory." → ?
* "I want to upload my tokenizer so a teammate can use it by name." → ?

If you can route those five jobs, you understand the ecosystem. Then find out what "Auto" means in `AutoTokenizer` (which file does it read to decide what class to build?).

### Exercise 1.1 — load tokenizers

**Pattern:** a library often has a class with a method that *creates* an object from a name. You have seen the pattern:

```python
# generic pattern:   Class.from_something(argument)
Tokenizer.from_file("models/x/tokenizer.json")       # builds a Tokenizer from a file
```
Here the class is `AutoTokenizer` and the method is the one named in the hint (it starts with `from_`). The argument is the model name, a **string**. The loop in the exercise already runs over the three names; you fill in the call. Check afterwards with `type(tok).__name__`.

### Exercise 1.2 — interrogate a tokenizer

You need four things from each tokenizer. In the exercise they are attributes (no brackets) except one that is a function (`len(...)`). A way to see which is which: in a scratch cell run `dir(tok)` and look for the names given in the hints.

Two sizes are asked on purpose: **compare them for each model**. When they differ, ask what could be *in* one but not in the other — the hint tells you the second one counts "everything including added tokens". Write down which models show a difference; Part G asks you to choose one of the two as "vocabulary size", and you must justify the choice.

### Exercise 1.3 — look inside

A "fast" tokenizer is a *wrapper* around a `tokenizers.Tokenizer`. The wrapped object is an attribute named in the hint; its four components are attributes too. Remember §3.3: **a component can be `None`**. Print the class name only if it exists.

After this exercise, **draw the five-box pipeline** (§4.2) for each of the three models on paper, writing which model/normalizer/pre-tokenizer/decoder each one has. You will use this drawing again.

### Questions of the notebook

Answer them using what you printed. For "what makes a tokenizer fast" think about *what language the fast one is written in* and what extra information it can give you (offsets). For "what configuration files" use the Hub listing cell.

---

## 7. Notebooks 02 and 03 — encoding, decoding, vocabularies (Part C)

### 7.1 The three levels of conversion

```
text ──tokenize──► tokens (strings)
tokens ──convert_tokens_to_ids──► ids (integers)
text ──encode──► ids    (the whole trip, and it may ADD special tokens!)
ids ──decode──► text
```
Try them on the toy tokenizer wrapped in the Transformers API (an object like the ones you get from `AutoTokenizer`):

```python
from transformers import PreTrainedTokenizerFast
fast = PreTrainedTokenizerFast(tokenizer_file="demo_tok.json", unk_token="[UNK]", pad_token="[PAD]")

fast.tokenize("the cat flew")                         # ['the', 'cat', '[UNK]']
fast.convert_tokens_to_ids(fast.tokenize("the cat flew"))   # [2, 3, 0]
fast.encode("the cat flew")                           # [2, 3, 0]
fast.decode(fast.encode("the cat flew"))              # 'the cat [UNK]'
```

### Exercise 2.1 — the round trip

Four calls, four methods. The hints give their names; the pattern is always `tok.method(thing)`. After running, **compare `text` with `s`**. If they differ, note *how*: this is the start of the explanation the notebook asks for.

### Exercise 2.2 — `encode` and special tokens

Some tokenizers add framing tokens (`[CLS] … [SEP]`) when you call `encode`. The hint names the argument that switches this off; it is a **keyword argument** with a boolean value. **Predict before running:** will the default version be longer, shorter or the same? By how many ids? Then run and look at `convert_ids_to_tokens(ids_default)` to *see* the added tokens.

This is one of the most important facts of the assignment: in Part G you will count tokens, and you must count them **without** the extras, otherwise one tokenizer looks worse than another for a silly reason.

### Exercise 2.3 — batches, padding, attention masks

Calling the tokenizer *like a function* on a **list of texts** processes a batch. Sentences of different lengths cannot be stacked in a rectangle, so shorter ones are filled with the padding token; the **attention mask** tells the model which positions are real (1) and which are filler (0). Toy example with the demo tokenizer (pad id = 1):

```python
b = fast(["the cat", "the cat sat on the mat"], padding=True)
b["input_ids"]        # [[2, 3, 1, 1, 1, 1], [2, 3, 4, 0, 2, 0]]
b["attention_mask"]   # [[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]]
```
(The `0`s in the second row of ids are unknown words, `[UNK]`; the `1`s in the first row are padding. Same number, different meaning — the **mask** is how you tell them apart!)

In your exercise: fill the list of two sentences and the padding argument (named in the hint), then check the lengths.

### The five sentences, then the "unknowns" experiment

The notebook gives you the loop. Your job is to **look** at the output and explain every difference between original and decoded text. Sort the differences into causes:

| Cause | Question to ask |
|---|---|
| the normalizer changed the text | did capitals/accents disappear? |
| the decoder re-inserted spaces or changed spacing | is the spacing different? |
| framing tokens | are there `[CLS]`/`</s>` in the output? |
| unknown tokens | is something replaced by an unknown token or missing? |

For each mismatch name the **station of the pipeline** that is responsible (§4.2). For the Chinese/emoji/Arabic experiment, look at three things per tokenizer: is there an unknown token? how many tokens? can you recover the original? You will find the three philosophies of §4.3.

### Notebook 03 — vocabularies

**Exercise 3.1 — extract and invert a vocabulary.** The vocabulary is returned as `{token: id}` by a method named in the hint. To invert a dictionary use a dict comprehension (§2.4):

```python
{i: t for t, i in vocab.items()}
```
The blank in the exercise is the *whole* comprehension body (`key: value for ... in ...`). Check that its length matches the original.

**Exercise 3.2 — lookups.** Use `dict.get` (safe, returns `None`) for token → id. Try several *spellings* of the same word: GPT-2 writes a word with a leading space as `Ġword`, T5 as `▁word`, BERT writes continuation pieces as `##piece`. Find out which marker means "starts a new word" and which means "continues the previous one" — they are **not** the same logic.

**Exercise 3.3 — multi-piece words.** A list comprehension with a condition (§2.4): keep words for which `len(tok.tokenize(word))` is *more than one*. Test it with a word you are sure is split and one you are sure is not.

Questions: *Are ids meaningful rankings?* Check with data: look at a few very common words and see whether their ids are small. Does the answer depend on the tokenizer? (GPT-2 ids come from the order of learning merges; BERT's from the way its vocabulary file was built; do your own investigation.)

---

## 8. Notebooks 04 and 05 — normalization and pre-tokenization (Part D)

### 8.1 Normalization (notebook 04)

**The comparison table.** The notebook tokenizes eight inputs with the three tokenizers. For each *pair of inputs that differ only by capitals, accents or punctuation*, ask: **which tokenizers return identical tokens?** Those have a normalizer that erases that difference.

**Exercise 4.1 — ask a tokenizer for its normalizer.** Reach it through the backend tokenizer (like exercise 1.3) and call the method named in the hint. Remember `None` (§3.3).

**Exercise 4.2 — compose normalizers.** Normalizers are small objects; a `Sequence([...])` runs them **in order**. Pattern with a different job:

```python
from tokenizers.normalizers import NFD, StripAccents, Lowercase, BertNormalizer, Sequence

Sequence([NFD(), StripAccents(), Lowercase()]).normalize_str("Crème BRÛLÉE")      # 'creme brulee'
BertNormalizer(lowercase=True, strip_accents=True).normalize_str("Crème BRÛLÉE")  # 'creme brulee'
```
(That is the same idea as your exercise, so now try a *different* order in a scratch cell: what happens if `StripAccents()` comes **before** `NFD()`? That tells you why the order matters, and the walkthrough question "why `NFD` first?" will be easy.)

After the table cell, look at the Arabic example: what do some normalizers do to Arabic diacritics? Think about whether "strip accents" could harm a language where marks distinguish words.

### 8.2 Pre-tokenization (notebook 05)

A pre-tokenizer makes the *first rough cut* of the text. The main algorithm (BPE etc.) then works **inside** each piece and can **never merge across a boundary**.

Compare four pre-tokenizers on a *different* string:

```python
s = "e-mail: ana@x.com 2024!"
Whitespace().pre_tokenize_str(s)
# ['e', '-', 'mail', ':', 'ana', '@', 'x', '.', 'com', '2024', '!']       (the notebook shows pairs (piece, offsets); I list the pieces)
WhitespaceSplit().pre_tokenize_str(s)
# ['e-mail:', 'ana@x.com', '2024!']
BertPreTokenizer().pre_tokenize_str(s)
# ['e', '-', 'mail', ':', 'ana', '@', 'x', '.', 'com', '2024', '!']
ByteLevel(add_prefix_space=False).pre_tokenize_str(s)
# ['e', '-', 'mail', ':', 'Ġana', '@', 'x', '.', 'com', 'Ġ2024', '!']
```
Look carefully:

* `WhitespaceSplit` cuts **only on spaces**: punctuation stays glued (`'e-mail:'`).
* `Whitespace` and `BertPreTokenizer` also cut on punctuation — and **drop the spaces**.
* `ByteLevel` cuts similarly but **keeps the space as part of the next piece** (`Ġana`: `Ġ` is "a space"). Nothing is dropped, which is why byte-level tokenizers can reproduce the text exactly.

Each result item is really `(piece, (start, end))`. The offsets tell you where the piece sits in the original string.

**Exercise 5.1:** instantiate the four classes (their names are in the hint; each is created with brackets, e.g. `Name()`; the byte-level one takes the argument shown in the hint) and call `.pre_tokenize_str(text)` on each.

**Exercise 5.2:** `Sequence([...])` also exists for pre-tokenizers (import it from the pre-tokenizers module; the exercise's import line already does it). Pattern:

```python
Sequence([WhitespaceSplit(), Punctuation(), Digits(individual_digits=True)]).pre_tokenize_str("a-b 42")
# ['a', '-', 'b', '4', '2']
```
That is the same idea as the exercise on a different sentence. In your exercise the blanks are the three components.

**Questions.** Use your outputs: where does each treat punctuation, spaces, hyphens, digits? Why can `Whitespace` not be *lossless* ("Hello   world" with three spaces vs "Hello world")? Why does the pre-tokenizer change the **final** tokens even with the same model?

---

## 9. Notebook 06 — training your own tokenizers (Parts E and F)

This is the heart of the assignment. Notebook 06 has two phases: first the 🧩 exercises (you write the Hugging Face code), then, after you package the code in `src/` and the tests pass, the experiments with your functions.

### 9.1 What "training a tokenizer" means

Building a tokenizer has **two kinds of ingredients**:

| You **choose** them (design) | The trainer **learns** them from data |
|---|---|
| normalizer, pre-tokenizer, decoder, special tokens, target vocabulary size, `min_frequency` | the vocabulary (and, for BPE, the merge rules) |

So the recipe is always the same five steps:

```
1. create the model              Tokenizer( SomeModel(...) )
2. attach the stations           tokenizer.pre_tokenizer = ...   tokenizer.decoder = ...   (and normalizer)
3. configure a trainer           SomeTrainer(vocab_size=..., special_tokens=[...], ...)
4. train                         tokenizer.train(files, trainer)   or   tokenizer.train_from_iterator(texts, trainer=trainer)
5. use it                        tokenizer.encode(text).tokens / .ids / .offsets   and   tokenizer.decode(ids)
```

### 9.2 A complete worked example with a *different* model (word-level)

Everything in the exercises follows this same recipe. Here it is with the simplest possible model, a "word-level" tokenizer, trained on three sentences:

```python
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

# 1. the model
tok = Tokenizer(WordLevel(unk_token="[UNK]"))

# 2. a station
tok.pre_tokenizer = Whitespace()

# 3. the trainer
trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]"], show_progress=False)

# 4. train (from an iterator of strings: no file needed)
tok.train_from_iterator(["the cat sat", "the dog sat", "the cat ran home"], trainer=trainer)

# 5. use it
enc = tok.encode("the cat flew")
print(enc.tokens)             # ['the', 'cat', '[UNK]']
print(enc.ids)                # [2, 3, 0]
print(enc.offsets)            # [(0, 3), (4, 7), (8, 12)]
print(tok.decode(enc.ids))    # 'the cat'
print(tok.get_vocab())        # {'[UNK]': 0, '[PAD]': 1, 'the': 2, 'cat': 3, 'sat': 4, 'dog': 5, 'home': 6, 'ran': 7}
```
Observe:

* **Special tokens get the first ids**, in the order of the list (`[UNK]` = 0, `[PAD]` = 1). This is why the order of the list matters; the tests check it.
* Training from **files** is `tok.train(["file1.txt"], trainer)` — a *list of strings* (paths); from memory it is `train_from_iterator(texts, trainer=trainer)`.
* After training, `get_vocab_size()` gives the final size.

Now your exercises are "the same, with other ingredients".

### Exercise 6.1 — assemble the (untrained) BPE tokenizer

Recipe steps 1 and 2. The **model** is BPE (hint: its constructor takes the unknown token). Then two stations: a **byte-level pre-tokenizer** and the matching **decoder**. The only trap: two classes are called `ByteLevel` (one in `pre_tokenizers`, one in `decoders`). The exercise's import line renames the decoder as `ByteLevelDecoder` — use each in the right place.

What is "byte-level"? Remember §2.10: every character is turned into bytes, and **the 256 possible byte values** are the starting symbols. So the tokenizer can write *any* text, even in a script it has never seen.

Why must the decoder be byte-level too? Try this thought experiment: the tokenizer works on bytes, so tokens look like `Ã©` for `é`. Something has to turn `Ã©` back into `é`. That is the decoder's job. After you finish exercise 6.3, **temporarily comment out the decoder line** and decode a sentence with an accent to see the garbage.

### Exercise 6.2 — configure the trainer

Step 3. Read the hints: five named arguments (`vocab_size`, `min_frequency`, `special_tokens`, `initial_alphabet`, `show_progress`).

* `vocab_size`: the **target** number of tokens in the final vocabulary (not a promise: the corpus might not provide enough frequent pairs).
* `min_frequency`: a pair must occur at least this many times to be merged. With a small corpus, a high value stops training early.
* `special_tokens`: a list of strings. Order = ids.
* `initial_alphabet`: the symbols that must be in the vocabulary even if they never occur in the corpus. For byte-level BPE, that is the list of the **256 byte symbols**, provided by a method of the pre-tokenizer class (the hint tells you which). Think: if your corpus has no Arabic and you *forgot* it, what would happen when someone tokenizes Arabic?

### Exercise 6.3 — train, then inspect

Steps 4 and 5. Training takes a few seconds. Two details:

* with `train`, the first argument is a **list of strings** (not a `Path` object, not a bare string);
* afterwards `encode(text)` returns an `Encoding`; its three attributes are the ones in the toy example (`tokens`, `ids`, `offsets`).

Look at your result: which tokens look like whole words? which are single bytes? What does `Ġ` mean? Why did `è` become two characters? (Hint: §2.10.)

### Exercise 6.4 — assemble the WordPiece tokenizer

Same recipe with **four** stations. The names are in the hint: a WordPiece model, a "Bert" normalizer, a "Bert" pre-tokenizer and a WordPiece decoder with `prefix="##"`.

The normalizer is the new station. It makes WordPiece **lossy**: capitals and accents disappear *before* the model sees the text. Predict: what will `decode(encode("Café"))` return?

How does WordPiece cut a word? It takes the **longest piece** from its vocabulary that matches at the start, then continues with the rest; continuation pieces carry the prefix `##`. If at some position **no piece matches**, the whole word becomes the unknown token.

### Exercise 6.5 — WordPiece trainer + `train_from_iterator`

Same as 6.2 but for WordPiece (the trainer class name appears in the hint, plus `continuing_subword_prefix`). Notice that this time you train **from an iterator of strings** (`lines`), exactly like the worked example of §9.2: no files.

### Exercise 6.6 — a post-processor

The fourth station: adding `[CLS] … [SEP]`. A "template" says where the sentence goes (`$A` means "the sentence"). You need the **ids** of the special tokens, obtained with `token_to_id`. After the exercise run both versions (with and without `add_special_tokens=False`) and compare — you saw this behaviour in exercise 2.2 on a pretrained BERT.

### Exercise 6.7 — vocabulary overlap (Jaccard)

Two vocabularies as **sets** of tokens: the overlap is `size of intersection / size of union`. In Python, `a & b` is the intersection and `a | b` the union (§Assignment 01 guide §2.5). Toy check: `{1,2,3}` and `{2,3,4}` share 2 of 4 → 0.5.

The experiment trains each tokenizer three times: are the vocabularies identical every time? Don't assume — look. What does your result imply for how a tokenizer should be **published**?

### ⏸ Stop: package your code (see §10)

### Exercise 6.8 — save and reload

A tokenizer is useless if it dies when you close the notebook. Pattern:

```python
from pathlib import Path
Path("models/scratch").mkdir(parents=True, exist_ok=True)
tok.save("models/scratch/demo.json")                       # a method on the tokenizer
again = Tokenizer.from_file("models/scratch/demo.json")    # a CLASS method that rebuilds it
again.encode("the cat flew").ids == tok.encode("the cat flew").ids      # True
```
Both calls take a **string** path (convert a `Path` with `str(...)`). Then the notebook opens the JSON: list its top-level keys and ask, for each, "if this part were missing, how would the tokenizer behave differently?" That list is your answer to "why is saving only the vocabulary not enough?".

```python
import json
data = json.loads(Path("models/scratch/demo.json").read_text(encoding="utf-8"))
list(data.keys())
# ['version', 'truncation', 'padding', 'added_tokens', 'normalizer', 'pre_tokenizer', 'post_processor', 'decoder', 'model']
```
(This is the toy tokenizer; yours will have the same keys with different content.)

### Exercise 6.9 — wrap for Transformers

`PreTrainedTokenizerFast` turns your saved file into a tokenizer with the whole Transformers toolbox: padding, attention masks, `save_pretrained`, and loading with `AutoTokenizer`.

```python
from transformers import PreTrainedTokenizerFast
fast = PreTrainedTokenizerFast(tokenizer_file="models/scratch/demo.json", unk_token="[UNK]", pad_token="[PAD]")
fast.special_tokens_map     # {'unk_token': '[UNK]', 'pad_token': '[PAD]'}
len(fast), fast.is_fast     # (8, True)
```
Key idea: the raw tokenizer knows **how to cut**, but not **which token is the padding token**; the wrapper is where you declare it. Declare **only the special tokens your tokenizer actually has** (your BPE has `[UNK] [PAD] [BOS] [EOS]`).

Then `save_pretrained(folder)` writes a folder, and `AutoTokenizer.from_pretrained(folder)` reads it back — the same call you used for GPT-2, now on **your** tokenizer.

### The experiments after packaging

* **Requested vs final vocabulary size** (a table): read where the final is *smaller* than the requested and find what they have in common.
* **20 sentences, both tokenizers:** which differences come from the normalizer, which from the algorithm, which from unseen characters? Note what WordPiece does with Arabic and who "wins" on token count — and whether the win means anything.
* **Reproducibility:** report what you found in numbers.

---

## 10. Packaging the code in `src/`

An exercise uses **constants** (`2000`, `"[UNK]"`, a specific file). A function uses **parameters**. To convert:

1. Copy your working lines from the notebook into the function body.
2. Replace constants by parameters.
3. `return` the object at the end.
4. Run the tests of that file.

Example of the transformation on the toy tokenizer:

```python
# exercise version (constants)
tok = Tokenizer(WordLevel(unk_token="[UNK]"))
tok.pre_tokenizer = Whitespace()

# src version (parameters, return)
def build_wordlevel_tokenizer(unk_token="[UNK]"):
    tok = Tokenizer(WordLevel(unk_token=unk_token))
    tok.pre_tokenizer = Whitespace()
    return tok
```

### Which exercise goes where

| Exercise(s) | → file / function |
|---|---|
| 6.1 | `src/train_bpe.py` → `build_bpe_tokenizer` |
| 6.2 | `make_trainer` |
| 6.3 (+ iterator idea of 6.5) | `train_bpe`, `train_bpe_from_texts` |
| 6.4 | `src/train_wordpiece.py` → `build_wordpiece_tokenizer` |
| 6.5 | `make_trainer`, `train_wordpiece`, `train_wordpiece_from_texts` |
| 6.8 | `src/tokenizer_io.py` → `save_tokenizer`, `load_tokenizer` |
| 6.9 | `to_transformers` |
| (new) | `find_mismatches` — compares two tokenizers on a list of sentences |

Details to watch (all of them are checked by the tests):

* `train_bpe(files, ...)`: `files` may be **one path or a list**; convert paths to strings; the **first special token** is the unknown token of the model; return the trained tokenizer.
* Default special tokens: `["[UNK]", "[PAD]", "[BOS]", "[EOS]"]` (BPE) and `["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]` (WordPiece).
* `save_tokenizer` must **create missing folders** and **return the path**.
* `to_transformers` must pass **only the special tokens that are not `None`** to the wrapper.
* `find_mismatches(before, after, sentences)` returns **one dictionary per sentence that differs** (ids *or* decoded text), with the keys listed in its docstring; an empty list means "identical behaviour".

Run, in this order:

```bash
pytest tests/test_bpe.py -x
pytest tests/test_wordpiece.py -x
pytest tests/test_tokenizer_io.py -x
```
You can also train from the command line once the functions exist:

```bash
python -m src.train_bpe --files data/processed/train_corpus.txt --out models/bpe/tokenizer.json
python -m src.train_wordpiece --files data/processed/train_corpus.txt --out models/wordpiece/tokenizer.json
```

---

## 11. Writing your own tests

The assignment requires **at least five tests of your own** in `tests/test_my_tokenizers.py`. A *test* is a function whose name starts with `test_` and that uses `assert`:

```python
# tests/test_example.py     (a toy example, NOT your file)
def double(x):
    return 2 * x

def test_double_of_three_is_six():
    assert double(3) == 6

def test_double_of_zero_is_zero():
    assert double(0) == 0
```
Run it with `pytest tests/test_example.py`. A test **passes** if no `assert` fails.

Good tests are small and named after the behaviour. For your saved tokenizers think of:

1. the saved file **exists and loads**;
2. a known sentence **encodes to a non-empty list of ids**;
3. `decode(ids)` returns what you **expect** — careful: the expected text is *not always identical to the input* (lossy normalizers!). Decide what is right for each tokenizer;
4. each **special token has the id** you declared;
5. encoding is **identical after save → reload**.

Extra ideas: a *negative* test (something your tokenizer must **not** do, e.g. produce an unknown token on a given input); a test that loads the tokenizer through `PreTrainedTokenizerFast`.

Tip: use `pytest.fixture` or a helper function at the top of the file to load each tokenizer once. Use the paths `models/bpe/tokenizer.json` and `models/wordpiece/tokenizer.json`, relative to the project folder.

---

## 12. Notebook 07 — the multilingual benchmark (Part G)

You now compare **five tokenizers** (GPT-2, BERT, FLAN-T5, your BPE, your WordPiece) on **300 sentences** in English, French, Modern Standard Arabic and Algerian Darija. Think of it as a **fair experiment**: same inputs, same rules for everybody.

### 12.1 The rules of fairness

1. **Same sentences** for all tokenizers.
2. **No special tokens counted** (`add_special_tokens=False`): remember exercise 2.2.
3. **The same definition of "vocabulary size"**: you chose it in exercise 1.2.
4. **Totals, not averages of ratios** for characters per token (§2.6).
5. **Read the unknown-token rate first**: a tokenizer that gives up on words can have *short* sequences and still be bad.
6. **Training data and test data are separate** (a test checks it).

### 12.2 The two APIs problem (exercise 7.1 and the adapter)

Your tokenizers (`tokenizers.Tokenizer`) and the pretrained ones (Transformers) have different method names for the same thing:

| Question | raw `tokenizers.Tokenizer` | Transformers tokenizer |
|---|---|---|
| token strings | `tok.encode(text, add_special_tokens=False).tokens` | `tok.tokenize(text)` |
| ids | `tok.encode(text, add_special_tokens=False).ids` | `tok.encode(text, add_special_tokens=False)` |
| vocabulary size | `tok.get_vocab_size(with_added_tokens=True)` | `len(tok)` |
| id of a token | `tok.token_to_id(token)` | `tok.convert_tokens_to_ids(token)` |

An **adapter** is a small class that hides the difference: the rest of your code calls `adapter.tokenize(text)`, `adapter.encode(text)`, `adapter.vocab_size`, `adapter.unk_id`, and the adapter decides which call to make. This table is exactly what its methods contain.

**Exercise 7.1** is the adapter in miniature: count the tokens of one sentence through both APIs.

### 12.3 What you compute for each sentence

| column | meaning |
|---|---|
| `n_chars` | number of characters: `len(sentence)` |
| `n_bytes` | number of UTF-8 bytes: `len(sentence.encode("utf-8"))` |
| `n_words` | number of words: `len(sentence.split())` |
| `n_tokens` | number of tokens produced |
| `n_unk` | how many of them are the unknown token |

`evaluate_sentences` builds one row per sentence (a list of dictionaries turned into a DataFrame). The unknown token's id comes from the adapter; if a tokenizer has **no** unknown token, `n_unk` is 0 — handle `None`.

### 12.4 Turning rows into metrics (exercise 7.2 and `summarize`)

A worked example with toy numbers (five sentences, two languages — *not* your data):

```python
df = pd.DataFrame({"lang":["en","en","fr","fr","fr"], "tokens":[10,12,20,22,30], "chars":[40,50,60,66,90]})

df["tokens"].mean()                               # average tokens per sentence      -> 18.8
df["tokens"].max()                                # maximum                          -> 30
df["chars"].sum() / df["tokens"].sum()            # characters per token (totals)    -> 3.255...
df.groupby("lang")["tokens"].mean()               # per language                     -> en 11.0, fr 24.0
```
`summarize(results, group_cols=["tokenizer", "language"])` does this for every **group** (each pair tokenizer × language). A loop over `results.groupby(group_cols)` gives you each group as a small DataFrame `g`; compute the metrics on `g` and collect the rows. The column names and definitions are listed in the docstring; the tests check them on tiny examples (with answers computed by hand), so **compute them by hand once on paper** first.

**Exercise 7.2** asks you to compute four of these metrics by hand for one tokenizer; the check then compares them with **your own** `summarize`. If they disagree, one of them is wrong — find which.

### 12.5 Timing (exercise 7.3)

Pattern (§2.8): one warm-up pass, then a few timed passes, then the mean. You write the core of `time_encoding`-style code; the check only verifies it is plausible. In the report state: your machine, how many repeats, one sentence at a time (not batched).

### 12.6 Tables and figures

* The **overall table**: one row per tokenizer (vocabulary size, average tokens, max tokens, characters per token, unknown rate, time).
* **One table per language**: does the ranking change by language?
* **Parity** (given function): for each tokenizer, `tokens per word (language) / tokens per word (English)`. A value of 4 means "this language costs 4× more tokens per word than English".
* **Exercise 7.4:** turn the long table into a wide one with `pivot(index=..., columns=..., values=...)`, then `.plot(kind="bar")`. Use `.loc[order]` so every figure lists tokenizers in the same order. The other figures (distributions, characters per token, vocabulary sizes) are provided so you can focus on interpretation.

### 12.7 How to interpret: claim → number → mechanism → caveat

For every analysis question:

1. **Claim:** "Tokenizer X is the most compact for French."
2. **Number:** the exact cell(s) of your table.
3. **Mechanism:** *why* — use the pipeline (§4.2) and the three philosophies (§4.3): training data? byte fallback? unknowns? vocabulary size?
4. **Caveat:** what the number does **not** show (small dataset, one author, whitespace words, one training domain).

Questions to keep asking while you read:

* Is the tokenizer with the fewest tokens automatically the best? What would you also need to know?
* Is a bigger vocabulary always better? Find evidence in your own table.
* Is Darija more or less costly than MSA? Give **two** possible explanations and say what extra data would let you choose.
* Why does sequence length matter downstream? (Think about what a model must do for every token, and what happens to the cost when the sequence gets four times longer.)

---

## 13. The report and the final challenge

### 13.1 The report as an argument

Not a diary of what you did, but an **argument supported by numbers**:

1. *Question:* how do tokenizer design and training data shape efficiency across languages?
2. *Method:* the pipeline (§4.2), your data, your metrics, your fairness rules.
3. *Findings:* three or four statements, each backed by a table or figure.
4. *Explanation:* mechanisms (normalization, pre-tokenization, model, decoder, training data).
5. *Limits:* be specific ("50 Darija sentences by a single author; whitespace words as unit" is better than "small dataset").

Sentence starters: "We observed that … (Table X)". "This happens because …". "A limitation is …". Use `report/REPORT_TEMPLATE.md`. Every number must come from a file in `results/`; include library versions and commands in the appendix.

### 13.2 The final challenge (1–2 pages)

You must design the tokenizer for a model in English, French, MSA and Darija with a budget of **32 000 tokens** — you do **not** train it. Treat it as a short research proposal and reuse **your own measurements** as evidence. For each of the five prompts of the handout:

| Prompt | How to think |
|---|---|
| which implementation? | recall what happens to a script not seen in training under each design (§4.3) |
| how to distribute training data? | the share of the vocabulary roughly follows the share of the data: compare "proportional to size", "equal per language", "something in between" and say what each does to a language with little text |
| how to stop English from dominating? | what can you change *before* training (data) and *after* (checking and adjusting)? |
| how to measure fairness? | pick a metric (e.g. parity, unknown rate, bytes per token) and a reference language, and justify — the choice of reference changes the numbers |
| vocabulary size vs sequence length? | use the trend in your own results |

Finish with **an experiment you could actually run**: what you vary, what you measure, what result would change your mind.

---

## 14. Troubleshooting FAQ

**"`pytest: command not found`."** Your virtual environment is not active. Activate it and run `pip install -r requirements.txt`.

**"`ModuleNotFoundError: No module named 'src'`."** Run `pytest` from the main project folder; in notebooks the first cell sets the path — run it first.

**"I changed `src/…` but the notebook behaves as before."** Restart the kernel (**Kernel → Restart**) and run again.

**"`AutoTokenizer.from_pretrained` fails with `OSError`."** No internet, a misspelled name, or the Hub is blocked. Check the name letter by letter (`google/flan-t5-small`), your connection, and try again. Tests that need the Hub are skipped automatically; the notebooks 01–04 and 07 need the models at least once.

**"My trained tokenizer has fewer tokens than I asked for."** Normal when the corpus does not contain enough frequent pairs: the requested size is a **target**. The notebook table shows when.

**"Decoded text is full of `Ã©`."** The decoder is missing or wrong for a byte-level tokenizer.

**"WordPiece decode gives `play ##ing`."** The WordPiece decoder is missing.

**"Special token ids are not 0, 1, 2…"** Check the order of the list given to the trainer.

**"My two WordPiece trainings give different vocabularies."** That is real: this trainer is not perfectly reproducible. It is the reason to **save** the trained tokenizer and evaluate *that file*.

**"`TypeError ... PosixPath`."** A `Path` object where a string is needed: use `str(path)`.

**"The benchmark numbers differ from my classmate's."** Pretrained models should give the same numbers; your WordPiece may differ slightly (see above). Compare using the **saved** `tokenizer.json` files.

**"Is it okay to use an AI assistant?"** To *explain* an error message or a concept: yes (cite it). To **write the exercise code for you**: no — you will be asked to explain every line, and the check cells and tests cannot save you in an oral discussion.

---

## 15. Glossary

| Word | Meaning |
|---|---|
| **token / id / vocabulary** | piece of text / its integer number / the table token ↔ id |
| **special token** | control symbol: `[UNK] [PAD] [BOS] [EOS] [CLS] [SEP] [MASK]` |
| **unknown token (`[UNK]`)** | what a tokenizer emits when it has no way to represent a piece |
| **normalizer** | station that cleans text before splitting (lowercase, accents, Unicode form) |
| **pre-tokenizer** | station that makes the first rough split; the model cannot merge across its boundaries |
| **model** | the subword algorithm: BPE, WordPiece, Unigram |
| **post-processor** | station that adds framing tokens like `[CLS]`/`[SEP]` |
| **decoder** | station that turns tokens back into text |
| **trainer** | object that learns the vocabulary from text |
| **byte-level** | working on UTF-8 bytes: any text can be written, nothing is ever unknown |
| **`Ġ`, `▁`, `##`** | markers: "a space before this" (GPT-2), "a space before this" (SentencePiece/T5), "continuation of the previous piece" (WordPiece) |
| **fast tokenizer** | Rust-backed tokenizer with offsets and batch speed |
| **offsets** | positions `(start, end)` of each token in the original text |
| **attention mask** | list of 1/0 telling a model which positions are real and which are padding |
| **padding** | filling short sequences with `[PAD]` so a batch is rectangular |
| **`tokenizer.json`** | file that stores the whole tokenizer pipeline |
| **adapter** | small class giving two different APIs one interface |
| **fertility** | average number of tokens per word |
| **characters per token** | total characters / total tokens (higher = more compact) |
| **parity** | cost in a language divided by cost in a reference language |
| **DataFrame** | pandas table; `groupby` and `pivot` reshape it |
| **unit test** | small automatic check of one behaviour; passes if no `assert` fails |

---

## 16. Plan and final checklist

### A realistic plan for a beginner (about 16 hours)

| Day | Work | Done when |
|---|---|---|
| 1 | Setup (§1), survival kit (§2), exploring tools (§3), `pytest` once | you can load a notebook and run a test |
| 2 | Notebook 01 (ex. 1.1–1.3) + Part A answers | all ✅ |
| 3 | Notebooks 02–03 (ex. 2.1–3.3) | all ✅ |
| 4 | Notebooks 04–05 (ex. 4.1–5.2) | all ✅ |
| 5–6 | Notebook 06 ex. 6.1–6.7, then `src/train_bpe.py` and `train_wordpiece.py` | `test_bpe.py`, `test_wordpiece.py` green |
| 7 | Notebook 06 ex. 6.8–6.9, `src/tokenizer_io.py`, your own tests | `test_tokenizer_io.py` green, ≥ 5 own tests |
| 8–9 | `src/evaluation.py`, notebook 07 exercises | `test_evaluation.py` green |
| 10 | Run notebook 07 fully, tables and figures, analysis | `results/` filled |
| 11–12 | Report + final challenge, restart & run all, checklist | submitted |

### Final checklist

**Code**
- [ ] All 26 🧩 exercises completed: no `____` left, all ✅ checks pass.
- [ ] `pytest` green (pretrained tests may be skipped offline; run them once online).
- [ ] `tests/test_my_tokenizers.py` has at least five tests, all passing.
- [ ] Provided tests untouched.

**Experiments**
- [ ] Notebooks 01–07 run top to bottom (**Restart & Run All**) without errors.
- [ ] `models/bpe/tokenizer.json` and `models/wordpiece/tokenizer.json` committed; they are the ones you evaluated.
- [ ] `results/` contains the sentence-level CSV, the comparison tables and four figures.

**Understanding** — you can explain, in your own words, without notes:
- [ ] the five stations of a tokenizer and what each does to one example;
- [ ] why saving only the vocabulary is not enough;
- [ ] why a short sequence is not automatically a good tokenization;
- [ ] the difference between "the tokenizer gave up and said so" and "the tokenizer succeeded at a higher cost";
- [ ] one thing in your benchmark that could be wrong.

**Report**
- [ ] Numbers come from `results/`; data sources and language distribution documented; final challenge included; sources cited.

---

*You do not need to remember every class name. You need to know how to explore an object, try a small example, read an error, and check your result. That is what working with real machine-learning libraries looks like.*