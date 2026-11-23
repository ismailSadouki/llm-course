# Assignment 01 — Beginner's Walkthrough
### Text Preprocessing and Tokenization — step by step, for people who are new to programming

> **Who this is for.** You are new to Python, or you know a little but feel lost when you open a notebook with `____` blanks. This guide goes slowly. It explains the Python you need, the ideas behind each part, and *how to think* when you get stuck.
> **What it does not do.** It does not give you the answers to the graded exercises or questions. It gives you **worked examples on different problems** (words like "banana", numbers, tags) so that you learn the technique and then apply it to your own task.
>
> If you are comfortable with Python already, read `WALKTHROUGH.md` instead (shorter, more conceptual).

---

## Contents

0. Read this first (it is normal to feel lost)
1. Setting up your computer
2. A Python survival kit (everything you need, with tiny examples)
3. How to work: exercises, hints, errors, tests
4. The ideas, in plain words
5. Part A — Preprocessing, step by step
6. Part B — Character and word tokenizers
7. Part C — BPE from scratch (the big one)
8. Part D — BPE on real data
9. Part E — WordPiece and Unigram
10. Part F — Embeddings
11. The report
12. Troubleshooting FAQ
13. Glossary
14. Plan and final checklist

---

## 0. Read this first

### 0.1 It is normal to feel lost

This assignment asks you to build things that people use inside ChatGPT-like systems. Nobody builds them on the first try. Professional programmers spend most of their time reading error messages and trying small experiments. The goal of the assignment is that **you** understand the pipeline, not that you type fast.

Three rules that make everything easier:

1. **Work in tiny steps.** Never write 30 lines and then run them. Write 2–3 lines, run, look at the result, continue.
2. **Print everything.** If you don't know what a variable contains, `print` it. Most bugs disappear when you *look* at the data.
3. **Predict, then run.** Before running a cell, say out loud: "I expect to see …". If the result is different, you just learned something.

### 0.2 What you will actually do

The assignment has two kinds of work:

| | What it is | Where |
|---|---|---|
| 🧩 **Exercises** (21) | small puzzles: a few lines of code with `____` blanks you fill in. You get hints and an automatic ✅ check | the notebooks (`notebooks/*.ipynb`) |
| 🛠 **Functions** | you copy your working pieces into proper functions so that the *tests* can check them | `src/*.py` |

You do the exercise **first** (small, easy to check), then move the code into `src/` (bigger, tested).

### 0.3 How long will it take?

Plan about 12–15 hours if you are a beginner (the handout says 8–12 for experienced students). Spread it over a week or more. Do **not** start the day before the deadline. See §14 for a plan.

### 0.4 Where to ask for help

Ask early, and ask well: say **what you tried**, **what you expected**, **what happened** (copy the exact error message), and **your code**. "It doesn't work" cannot be answered; "I get `KeyError: 'w'` on line 4 of this cell, here is the cell" can.

---

## 1. Setting up your computer

You only do this once.

### 1.1 What is a terminal?

A terminal is a window where you type commands instead of clicking. On Windows use *PowerShell* or *Git Bash*; on macOS use *Terminal*; on Linux you know. Useful commands:

| Command | What it does |
|---|---|
| `cd folder_name` | go into a folder |
| `cd ..` | go back up one folder |
| `ls` (or `dir` on Windows) | list the files |
| `python --version` | check Python is installed (you need 3.10 or newer) |

### 1.2 Get the project

Your teacher gives you a zip file or a GitHub link.

```bash
# if it is a GitHub repository
git clone <the-link> assignment-01
cd assignment-01
```
If it is a zip: unzip it and `cd` into the folder (the one that contains `README.md`).

### 1.3 A virtual environment (a private box for this project's libraries)

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows
pip install -r requirements.txt
```
After activating, the terminal prompt shows `(.venv)`. **Every time you open a new terminal for this assignment, activate it again.**

### 1.4 Run the tests (and expect red)

```bash
pytest
```
You will see many failures like `NotImplementedError`. **That is the correct starting point**: those functions are not written yet. Your job is to turn the red into green, one file at a time.

### 1.5 Open the notebooks

```bash
jupyter notebook
```
A browser tab opens. Click `notebooks/`, then `01_preprocessing.ipynb`.

**How a notebook works**
* It is made of **cells**. Click a cell and press **Shift + Enter** to run it.
* Run cells **from top to bottom**. If you skip one, later cells may fail.
* The result of the last line appears under the cell.
* If something is weird: **Kernel → Restart & Run All** gives a clean start.
* Save with Ctrl+S (or Cmd+S). Jupyter also autosaves.

---

## 2. A Python survival kit

Everything below is used in the assignment. Type each example into a scratch cell (use **`Esc` then `B`** to add a cell below) and look at the output. The examples use words like "banana" on purpose: they are **not** your exercises.

### 2.1 Variables, strings and `print`

```python
word = "banana"
print(word)            # banana
print(len(word))       # 6
print(word[0])         # b      (counting starts at 0!)
print(word[1:])        # anana  (from position 1 to the end)
```
`word[1:]` is a **slice**: "everything starting at position 1". Slices will matter a lot (§7).

### 2.2 Lists

A list is an ordered collection you can change.

```python
letters = ["b", "a", "n"]
letters.append("a")        # add at the end   -> ['b', 'a', 'n', 'a']
print(letters[0], len(letters))
for letter in letters:     # loop over the items
    print(letter)
```
`list("banana")` turns a string into a list of its characters.

### 2.3 Tuples — a list you cannot change

```python
pair = ("a", "n")
print(pair[0], pair[1])
```
A tuple is written with round brackets. A one-element tuple needs a trailing comma: `("x",)`.
**Why do we care?** Tuples can be **dictionary keys**; lists cannot:

```python
{("a", "b"): 1}     # works
{["a", "b"]: 1}     # TypeError: unhashable type: 'list'
```
(That is why BPE words are stored as tuples.)

### 2.4 Dictionaries — a lookup table

```python
ages = {"Ali": 20, "Sara": 22}
print(ages["Ali"])              # 20
ages["Omar"] = 19               # add an entry
print(ages.get("Sara", 0))      # 22   (value if the key exists)
print(ages.get("Lina", 0))      # 0    (the default if it does not)
print(ages.get("Lina"))         # None (no default given)
for name, age in ages.items():  # loop over key and value together
    print(name, age)
```
`d["missing_key"]` raises a `KeyError`; `d.get("missing_key", default)` does not. This difference matters for unknown tokens.

### 2.5 Sets — "have I seen this before?"

```python
seen = set()           # an EMPTY set. ({} would make an empty dict!)
seen.add("a")
print("a" in seen)     # True
print("z" in seen)     # False
print("z" not in seen) # True
```
A set has **no order** and **no duplicates**. `set("banana")` is `{'b', 'a', 'n'}` in an arbitrary order — which is why we use `sorted(...)` when we need a stable order.

### 2.6 Loops and `enumerate`

```python
for i, letter in enumerate(["a", "b", "c"]):
    print(i, letter)          # 0 a / 1 b / 2 c
```
`enumerate` gives you the position **and** the item. `while` loops repeat while a condition is true:

```python
i = 0
while i < 3:
    print(i)
    i += 1                    # i = i + 1
```

### 2.7 Comprehensions — a loop in one line

```python
squares = [n * n for n in range(5) if n % 2 == 0]    # [0, 4, 16]
lengths = {w: len(w) for w in ["hi", "hello"]}        # {'hi': 2, 'hello': 5}
```
Read it like a sentence: "n times n, **for** every n in 0..4, **if** n is even". The same shape builds dictionaries (`{key: value for ...}`).

### 2.8 Functions

```python
def shout(text, times=2):
    """Return the text in capitals, repeated."""
    return (text.upper() + " ") * times

print(shout("hi"))          # HI HI
```
`def` defines it, parameters go in brackets, `return` gives the result back. A function that does not `return` gives `None`.
A `lambda` is a tiny anonymous function: `double = lambda x: x * 2`.

### 2.9 `zip` — walk two lists side by side (very important in BPE)

```python
list(zip("banana", "banana"[1:]))
# [('b','a'), ('a','n'), ('n','a'), ('a','n'), ('n','a')]
```
You pair every letter with **the next one**: that is how you get all neighbouring pairs of a sequence. Draw it on paper:

```
b a n a n a
  b a n a n a      <- the same word shifted by one
  ↓ ↓ ↓ ↓ ↓
(b,a)(a,n)(n,a)(a,n)(n,a)
```

### 2.10 `Counter` — counting things

```python
from collections import Counter
print(Counter("banana"))                    # Counter({'a': 3, 'n': 2, 'b': 1})

c = Counter()
for pair in zip("banana", "banana"[1:]):
    c[pair] += 1                            # works even if the key is new
print(c.most_common(2))                     # [(('a','n'), 2), (('n','a'), 2)]
```
Note `c[pair] += 1` and the **tie** between `('a','n')` and `('n','a')`: both are 2. Ties will matter in BPE.

### 2.11 Sorting and `min` with a `key`

`sorted` and `min` accept a `key` function that says "compare by this".

```python
sorted(["pear", "fig", "apple", "kiwi"], key=lambda w: len(w))
# ['fig', 'pear', 'kiwi', 'apple']   <- by length (ties keep the original order)

sorted(["pear", "fig", "apple", "kiwi"], key=lambda w: (len(w), w))
# ['fig', 'kiwi', 'pear', 'apple']   <- by length, then alphabetically
```
A **tuple** as the key means "compare the first element; if equal, compare the second; …".
To sort **largest first**, put a minus sign on numbers:

```python
scores = [("a", 2), ("b", 5), ("c", 5)]
sorted(scores, key=lambda t: (-t[1], t[0]))
# [('b', 5), ('c', 5), ('a', 2)]     <- highest score first; ties alphabetical
```
`min(...)` works the same way and returns the **smallest by that key**:

```python
counts = {"x": 3, "y": 1, "z": 1}
min(counts, key=lambda k: (counts[k], k))    # 'y'   (lowest count; tie -> alphabetical)
```
(Iterating over a dict gives its keys.)

### 2.12 Regular expressions (regex) — patterns for text

A regex describes a *kind* of text. The pieces you need:

| Piece | Means |
|---|---|
| `\d` | one digit |
| `\s` | one whitespace character (space, tab, newline…) |
| `+` | "one or more of the thing before" |
| `re.sub(pattern, replacement, text)` | replace every match |

```python
import re
re.sub(r"\d+", "#", "call 123 or 45 now")     # 'call # or # now'
```
Always write the pattern as a **raw string** `r"..."` so that `\d` stays `\d`.

### 2.13 Classes and objects (you only need to read them)

A class is a blueprint that bundles **data** and **functions** (called *methods*).

```python
class Basket:
    def __init__(self):            # runs when you create a Basket()
        self.items = []            # data stored on the object
    def add(self, item):           # a method
        self.items.append(item)
        return len(self.items)

b = Basket()
b.add("apple"); print(b.items)     # ['apple']
```
`self` means "this object". Inside a method you read and write the object's data with `self.something`. The tokenizers of the assignment are classes: `self.token_to_id` is data stored on the tokenizer.

### 2.14 Imports

```python
import re                                  # use as re.sub(...)
from collections import Counter            # use as Counter(...)
from src.bpe import BPETokenizer           # YOUR code in src/bpe.py
```

### 2.15 NumPy arrays (needed in Part F)

```python
import numpy as np
table = np.array([[10, 11], [20, 21], [30, 31]])
print(table.shape)            # (3, 2)   3 rows, 2 columns
print(table[[2, 0, 2]])       # rows 2, 0 and 2:  [[30 31] [10 11] [30 31]]
print(np.eye(3))              # 3x3 identity matrix (1 on the diagonal)
print(np.array([1, 0, 0]) @ table)   # [10 11]  matrix product
```
`table[[2, 0, 2]]` ("fancy indexing") picks **whole rows** by their numbers. `@` is matrix multiplication.

### 2.16 f-strings (nice printing)

```python
n = 3
print(f"I have {n} apples and {n * 2} pears")
```

---

## 3. How to work: exercises, hints, errors, tests

### 3.1 The method for a 🧩 exercise (follow it literally)

1. **Read the statement and the hints.** Hints name the functions you need.
2. **Say in plain words what the code must do.** Example: "for each word, count it".
3. **Write the steps as comments or on paper** (this is called *pseudo-code*):
   ```
   # for each pair of neighbours in the word
   #     add the word's frequency to that pair's counter
   ```
4. **Look at the code around the blank.** What kind of thing must go there (a number? a list? a function call)? The names before and after tell you.
5. **Try the idea in a scratch cell with a tiny example** (e.g. the word "banana").
6. **Fill the blank, run the cell, run the ✅ check cell.**
7. **If the check fails:** read the message. It tells you what it expected. Print your variables and compare.

### 3.2 Reading error messages

Python errors look scary but follow a pattern: read the **last line** first.

| Last line says | What it means | What to do |
|---|---|---|
| `NameError: name '____' is not defined` | a blank is still empty | fill it |
| `NameError: name 'x' is not defined` | you used a variable you never created (or skipped a cell) | run the cells above, check spelling |
| `SyntaxError: invalid syntax` | the line is not valid Python (missing bracket, colon, quote) | look at the line the arrow points to |
| `IndentationError` | wrong number of spaces at the start of a line | Python uses indentation as structure: align with its block |
| `KeyError: 'abc'` | you asked a dict for a key that is not there | use `.get(key, default)` or print the dict |
| `TypeError: unhashable type: 'list'` | you used a list where a tuple/string is needed (dict key, set item) | convert with `tuple(...)` |
| `TypeError: ... takes 2 positional arguments but 3 were given` | wrong number of arguments | check the function's definition / `help(function)` |
| `AttributeError: 'list' object has no attribute 'xyz'` | that kind of object has no such method | `dir(obj)` lists what exists |
| `AssertionError` (in a ✅ cell) | your code ran but the result is wrong | compare your result with what the check says |
| `NotImplementedError` | you are calling a `src/` function you have not written yet | write it |

### 3.3 `print` and `help` are your best friends

```python
print(type(x), x)     # what is it and what does it contain?
help(zip)             # documentation of a function
```
In Jupyter you can also type a function name followed by `?`, e.g. `Counter?`.

### 3.4 Reading a failing test

```bash
pytest tests/test_bpe.py -x
```
`-x` stops at the **first** failure so you are not drowned. Read:

1. the **test name** — it describes what should happen;
2. the line with `assert` — it shows *expected* vs *actual*;
3. then copy that case into a notebook cell and print intermediate values.

Fix one failure, rerun the **same file**, repeat. Do not try to fix everything at once.

### 3.5 From exercise to `src/`

An exercise uses *constants* ("the toy corpus", "the pair `('a','n')`"). A function in `src/` uses *parameters*. To convert:

1. Copy the working lines into the function body.
2. Replace the constants by the parameter names.
3. Put `return` at the end with the result.
4. Run the tests for that file.

Example of the transformation, on a different task:

```python
# exercise version (constants)
text = "a b a"
counts = Counter(text.split())

# src version (parameters, return)
def count_words(text):
    return Counter(text.split())
```

### 3.6 Save your progress (git, minimal)

```bash
git add .
git commit -m "Part A: exercises 1.1-1.3 done"
```
Commit every time something works. If you break something later, you can go back.

---

## 4. The ideas, in plain words

### 4.1 What is text to a computer?

A computer stores **numbers**. To store text, we agree on a numbering of characters: **Unicode**. Each character gets a **code point** (a number). Then **UTF-8** is one way to write those numbers as **bytes** (a way to save them in files).

```python
"ñ"                                  # 1 character, 1 code point
len("ñ".encode("utf-8"))             # 2 bytes
```
Surprise: some characters can be written in more than one way. `ñ` can be *one* code point (`0xf1`) or *two* (`n` followed by a "combining tilde", `0x6e 0x303`):

```python
import unicodedata
[hex(ord(c)) for c in "ñ"]                                    # ['0xf1']
[hex(ord(c)) for c in unicodedata.normalize("NFD", "ñ")]      # ['0x6e', '0x303']
```
They *look* the same on screen but are different strings for Python (`==` says `False`). **Normalization** chooses one standard way to write them.

### 4.2 What is a language model's input?

A model cannot read letters. The pipeline is:

```
"I love cats"  →  tokens  →  ids  →  vectors
                 ["I","love","cats"]   [7, 52, 311]   [[0.1, -0.3, …], …]
```
* **Tokenization** cuts the text into pieces (*tokens*).
* A **vocabulary** is a dictionary: token → number (*id*).
* An **embedding table** gives each id a list of numbers (a *vector*) that the model can compute with.

Think of a **library**: tokens are book titles, the vocabulary is the catalogue number for each title, and the embedding table is the shelf where you find the book by its number.

### 4.3 Three ways to cut text

| Idea | Example for "unhappiness" | Good | Bad |
|---|---|---|---|
| **characters** | u n h a p p i n e s s | tiny vocabulary, never "unknown" | very long sequences |
| **words** | unhappiness | short sequences | huge vocabulary; unseen words are unknown |
| **subwords (BPE)** | un ·happi·ness (illustration) | a compromise; can build unseen words | needs training |

You will measure these trade-offs yourself with numbers.

### 4.4 Cleaning versus destroying

Old-style cleaning lowercases everything, removes punctuation and removes common words. Modern language models are trained to **predict the next real word**, so they need the real text, with capitals, punctuation and small words. A good question to ask about every cleaning step: **"Can I undo this? If not, what did I lose?"**

---

## 5. Part A — Preprocessing, step by step

**Files:** `notebooks/01_preprocessing.ipynb`, `src/preprocessing.py`, tests: `pytest tests/test_preprocessing.py`.

### Exercise 1.1 — Unicode

**In plain words:** take a string that is written the "decomposed" way, turn it into the "composed" way, then check the number of bytes and the names of the characters.

**What you need from §2:** calling a function with arguments; strings; list comprehension.

**Hints in the notebook name three functions** in the `unicodedata` module and one string method. Try them first on a different character in a scratch cell:

```python
import unicodedata
unicodedata.name("ñ")                                  # 'LATIN SMALL LETTER N WITH TILDE'
unicodedata.normalize("NFC", "n\u0303") == "ñ"         # True?  Run it and see
"ñ".encode("utf-8")                                    # b'\xc3\xb1'
```
Note: `normalize` takes **two** arguments: the form *as a string* (`"NFC"`) and the text. In the exercise, the blank inside quotes stays inside quotes.

### Exercise 1.2 — Whitespace

**In plain words:** many spaces/tabs/newlines in a row → one space; no spaces at the ends.

**Technique:** `re.sub(pattern, replacement, text)`. In §2.12 you saw `\d+` for "one or more digits". Which similar piece means "one or more whitespace characters"? What should the replacement be? Then, at the end, `.strip()` removes spaces at both ends — chain it after the `re.sub(...)` call with a dot.

Test your pattern on `"a   b\tc"` in a scratch cell before the exercise.

### Exercise 1.3 — HTML (the longest one: take your time)

**Why not regex?** Run this in a scratch cell and look at the damage:

```python
import re
re.sub(r"<[^>]+>", " ", "This is <b>very</b>. Really.")     # 'This is  very . Really.'
```
There is a stray space before the full stop. We need to treat `<b>` (inline) differently from `<p>` (a block).

**The tool:** Python's `HTMLParser` reads an HTML string and *calls your methods* whenever it sees something:

* `handle_starttag(tag, attrs)` — when it sees `<p>`, `<b>`, …
* `handle_endtag(tag)` — when it sees `</p>`, `</b>`, …
* `handle_data(data)` — when it sees plain text.

Here is a **different** parser that only *counts tags*, to show how it works:

```python
from html.parser import HTMLParser
from collections import Counter

class TagCounter(HTMLParser):
    def __init__(self):
        super().__init__()            # let HTMLParser set itself up first
        self.counts = Counter()       # our own data
    def handle_starttag(self, tag, attrs):
        self.counts[tag] += 1

t = TagCounter()
t.feed("<p>Hi <b>x</b> <b>y</b></p>")
t.close()
print(t.counts)                       # Counter({'b': 2, 'p': 1})
```
`feed` gives it the text; Python then calls `handle_starttag` for you each time.

**Your exercise, in plain words** (pseudo-code):

```
parts = []          # list of text pieces we keep
skip_depth = 0      # are we inside <script> or <style>?  (0 = no)

when a start tag arrives:
    if it is script/style:   skip_depth goes up by 1
    elif it is a block tag:  add one space to parts
when an end tag arrives:
    if it is script/style:   skip_depth goes down by 1 (never below 0)
    elif it is a block tag:  add one space to parts
when text arrives:
    if skip_depth is 0:      add the text to parts
at the end: join all parts into one string
```
That is exactly the structure of the exercise. Match each line of pseudo-code to a blank. Test it on the three little examples printed by the exercise cell.

**Two tiny details.** `x += 1` means `x = x + 1`. `"".join(list_of_strings)` glues pieces together with nothing between them.

### Exercise 1.4 — Remove duplicates but keep the order

**Why not `list(set(docs))`?** Try it: a set forgets the order.

**Idea in words:** go through the documents one by one; remember the ones you have seen; keep a document only the *first* time you meet it.

Here is the same *idea* on a different problem — keep only the first word starting with each letter:

```python
seen = set()
firsts = []
for w in ["apple", "avocado", "banana", "blueberry", "cherry"]:
    if w[0] not in seen:
        seen.add(w[0])
        firsts.append(w)
print(firsts)        # ['apple', 'banana', 'cherry']
```
In your exercise you remember the **whole document** instead of its first letter.

### Exercise 1.5 — Collisions with pandas

You will meet `pandas`, a library for tables (a `DataFrame` is a table). `df.groupby("column")` groups rows that share the same value in that column. Read the hint, run the cell, and **print the intermediate objects** (`print(groups)`) to see what each step produces.

### After the exercises: packaging in `src/preprocessing.py`

| Exercise | → function in `src/preprocessing.py` |
|---|---|
| 1.1 | `normalize_unicode(text)` |
| 1.2 | `normalize_whitespace(text)` |
| 1.3 | `remove_html(text)` (put your class above the function) |
| 1.4 | `remove_duplicates(documents)` |

Then write:

* `preprocess_text(text)` — one document through the steps. First decide: *what should happen to `None`?* (the docstring tells you). Then call your three functions in a sensible **order** (the notebook asks you to explain it).
* `drop_empty(documents)` — keep only strings that are not blank.
* `preprocess_corpus(documents)` — clean each, drop empty, drop duplicates, and count what was removed (the dictionary of statistics).

Run `pytest tests/test_preprocessing.py -x` and fix failures one by one. Remember §3.4.

### The experiment (A.3)

For each pair of sentences, ask: **"If a model only saw the cleaned text, could it tell these two sentences apart?"** Fill the table, run the cells, and answer the discussion in your own words with the table as evidence.

---

## 6. Part B — Character and word tokenizers

**Files:** `notebooks/02_basic_tokenizers.ipynb`, `src/char_tokenizer.py`, `src/word_tokenizer.py`, tests: `pytest tests/test_tokenizers.py`.

### What a tokenizer object must do

```
train(texts)   →  fill the vocabulary (token → id and id → token)
encode(text)   →  list of ids
decode(ids)    →  text again
```
It is a **class**, so the data (`token_to_id`, `id_to_token`) lives on `self`. The constructor is already written for you.

### Special tokens in one picture

```
id 0 [PAD]   padding: fills short sentences so a batch has equal lengths
id 1 [UNK]   unknown: "I have no entry for this"
id 2 [BOS]   beginning of sequence
id 3 [EOS]   end of sequence
id 4, 5, 6…  the real characters / words
```

### Exercise 2.1 — Character vocabulary

**In words:** (1) find the different characters, (2) put them in a fixed order, (3) number them after the four special tokens, (4) also build the reverse table.

**Techniques:** `set(text)` gives distinct characters; `sorted(...)` orders them; `enumerate(...)` numbers things; and a dict comprehension builds a dictionary. A different example of exactly these techniques:

```python
fruits = ["fig", "apple", "fig", "pear"]
distinct = sorted(set(fruits))                              # ['apple', 'fig', 'pear']
to_id = {f: i for i, f in enumerate(["<none>"] + distinct)}
# {'<none>': 0, 'apple': 1, 'fig': 2, 'pear': 3}
to_fruit = {i: f for f, i in to_id.items()}                 # flip keys and values
```
Concatenate two lists with `+`: `["a"] + ["b"]` is `["a", "b"]`.

### Exercise 2.2 — Encode and decode

**Encode, in words:** for each character in the text, look up its id; if the character is not in the vocabulary, use the id of `[UNK]` instead.

The tool for "look up or use a default" is `dict.get(key, default)` (§2.4).

**Decode, in words:** for each id, find its token; skip special tokens; glue the characters together with `"".join(...)`.

Tip: a list comprehension can include an `if` at the end to skip things (§2.7).

### Exercise 2.3 — Count and order words

**In words:** count each token, keep those seen often enough, and order them: most frequent first, and alphabetical when tied.

* `Counter(...)` does the counting (§2.10);
* the **tuple key with a minus sign** orders "highest first, then alphabetical" (§2.11). Practise on the `scores` example above, then apply it.

### Exercise 2.4 — OOV rate

A fraction = (how many are unknown) / (how many in total). Count with `sum(1 for ... if ...)`. Handle the empty list (avoid dividing by zero).

### Packaging

| Exercise | → |
|---|---|
| 2.1 | `CharacterTokenizer.train` |
| 2.2 | `CharacterTokenizer.encode` / `decode` (and the word tokenizer's, with the provided `word_tokenize`) |
| 2.3 | `WordTokenizer.train` |
| 2.4 | `WordTokenizer.oov_rate` |

Remember: **always start `train` by resetting** the vocabulary (`self._reset()` is provided), otherwise training twice keeps old entries.

### The real-corpus experiments

You will run your tokenizers on Shakespeare and plot the OOV rate. Before running, **predict**: will the OOV rate on *new* text be bigger or smaller than on the training text? What happens to it when you demand that a word be seen more times to be kept?

---

## 7. Part C — BPE from scratch (the big one)

**Files:** `notebooks/03_bpe_toy.ipynb`, `src/bpe.py`, tests: `pytest tests/test_bpe.py -x`.
This part is 30 % of the grade. Don't rush; the idea is simple, the details need patience.

### 7.1 BPE in one story

Imagine you must invent short codes for a language. You look at all the text and notice the letters "t" and "h" appear next to each other all the time. So you create one new symbol "th". Now you look again at the text *with* "th" and find "th" + "e" is frequent: new symbol "the". You repeat. After a while, frequent chunks of text have become single symbols. **Each "glue these two symbols" decision is a merge rule. The ordered list of merge rules is the tokenizer.**

### 7.2 A complete example on a different corpus

Corpus: `hug hug hug pug pug pun pun bun`

**Step 0 — word table.** Write each word as its characters plus a marker `</w>` ("end of word"), and note how often it occurs:

```
3 x  h u g </w>
2 x  p u g </w>
2 x  p u n </w>
1 x  b u n </w>
```

**Step 1 — count neighbouring pairs, remembering each word's frequency.** In `h u g </w>` (3 times) the pairs are `(h,u)`, `(u,g)`, `(g,</w>)` — each counts **3**. Do that for every word and add up:

```
(u,g) 5     (g,</w>) 5     (p,u) 4     (h,u) 3     (u,n) 3     (n,</w>) 3   ...
```
`(u,g)` appears in `hug` (3×) and in `pug` (2×) → 3 + 2 = 5.

**Step 2 — pick the most frequent pair.** Two pairs tie at 5. We need a rule so everyone gets the same answer: **choose the smallest pair alphabetically.** Compare as tuples: `("g","</w>")` vs `("u","g")` — `"g"` comes before `"u"`, so `("g","</w>")` wins.

**Step 3 — merge it everywhere.** Replace every `g` followed by `</w>` with the single symbol `g</w>`:

```
3 x  h u g</w>
2 x  p u g</w>
2 x  p u n </w>
1 x  b u n </w>
```

**Step 4 — repeat** (count again with the new symbol, pick, merge…). A real run of 6 merges gives:

```
(g,</w>)  (u,g</w>)  (h,ug</w>)  (n,</w>)  (u,n</w>)  (p,ug</w>)
```
Every merge is recorded, **in order**.

**Step 5 — use the rules on a new word.** Take `"hugs"`. Start from its characters: `h u g s </w>`. Apply the rules in the order they were learned: `(g,</w>)` doesn't apply (there's an `s` after `g`)... and so on until no rule applies. The result is a mix of pieces: BPE can write a word it has never seen using pieces it knows. A character it has never seen simply stays as itself.

### 7.3 Do it by hand first

The notebook asks you to compute the first two merges of **your** toy corpus on paper. Do it **before** coding: it is the best investment of the assignment. Use a pencil; make the word table; count the pairs; choose; merge.

### 7.4 The six exercises, in plain words

For each, I give you the idea, the Python tool, and a *parallel* example on a different problem.

#### Exercise 3.1 — the word table

**Idea:** turn the text into `{ (symbols…): frequency }`.
**Tools:** `text.split()` (words), `Counter` (frequencies), `tuple("abc")` gives `('a','b','c')`, adding tuples: `("a","b") + ("!",)`.

```python
counts = Counter("to be or not to be".split())     # Counter({'to': 2, 'be': 2, 'or': 1, 'not': 1})
table = {tuple(w) + ("|",): f for w, f in counts.items()}
# {('t','o','|'): 2, ('b','e','|'): 2, ('o','r','|'): 1, ('n','o','t','|'): 1}
```
(Your marker is `"</w>"`, not `"|"`.)

#### Exercise 3.2 — count the neighbouring pairs, weighted

**Idea:** for every word, for every pair of neighbours, add the word's frequency to that pair's counter.
**Tools:** `zip(symbols, symbols[1:])` and a `Counter`.

Pseudo-code:

```
pairs = empty counter
for each (symbols, freq) in the table:
    for each (a, b) in the neighbours of symbols:
        pairs[(a, b)] += freq
```
Parallel example (no frequencies, one word): count neighbours in "banana":

```python
c = Counter()
for a, b in zip("banana", "banana"[1:]):
    c[(a, b)] += 1
print(c)       # Counter({('a','n'): 2, ('n','a'): 2, ('b','a'): 1})
```
Your version adds `freq` instead of `1`.

#### Exercise 3.3 — choose the best pair (with the tie rule)

**Idea:** biggest count wins; if tied, the alphabetically smallest pair wins.
**Tool:** `min(pairs, key=...)` with a tuple key. Think: "smallest key" must mean "biggest count" → use a **minus** on the count (§2.11), then the pair itself for ties.

Parallel: highest score, ties alphabetical:

```python
scores = {"b": 5, "c": 5, "a": 2}
min(scores, key=lambda k: (-scores[k], k))      # 'b'
```
The notebook includes a tie example with its own check: use it to verify your key.

#### Exercise 3.4 — merge a pair everywhere

**Idea:** go through a word's symbols from left to right. When the current symbol and the next one form the pair, write the merged symbol and **skip both**; otherwise copy the current symbol and move by one.

Here is the technique on a **single list** (your exercise does this for every word of a table):

```python
def merge_in_list(symbols, pair):
    out = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(pair[0] + pair[1])     # glue the two symbols
            i += 2                            # skip both
        else:
            out.append(symbols[i])            # keep this symbol
            i += 1                            # move by one
    return out

merge_in_list(list("banana"), ("a", "n"))      # ['b', 'an', 'an', 'a']
merge_in_list(list("aaaa"), ("a", "a"))        # ['aa', 'aa']
```
Study why `i < len(symbols) - 1` is there (what would happen at the last symbol without it?). Now adapt to a **table**: loop over `table.items()` (symbols and frequency), build the new symbols, turn them into a tuple, and put them in a new dict with the same frequency. If two different words become the same tuple, **add** their frequencies (`new.get(key, 0) + freq`). Never change the input table.

#### Exercise 3.5 — the training loop

**Idea:** repeat: count pairs → choose the best → remember it → merge it. Stop if there are no pairs.

```
merges = []
repeat N times:
    pairs = count the pairs of the table
    if there are no pairs: stop
    best  = the chosen pair
    merges.append(best)
    table = the merged table
```
This only **combines** the three functions you already wrote. If the check fails, print `best` at each step and compare it with your hand computation.

#### Exercise 3.6 — encode a new word

**Idea:** start from the word's letters plus `</w>`. Look at the neighbouring pairs of the current word. Among those that are in the list of merge rules, apply the one that was **learned first**. Repeat until none applies.

**Tools:**
* `merge_ranks = {pair: position}` built with `enumerate(merges)` (the "rank" is the order of learning);
* `candidates = [(rank, position), ...]` and `min(candidates)` gives the earliest rank (tuples compare element by element);
* to merge that pair inside the word, reuse your `merge_pair` on a one-word table.

Draw the example by hand with 2–3 merge rules before coding.

### 7.5 Packaging in `src/bpe.py`

| Exercise | → method of `BPETokenizer` |
|---|---|
| 3.1 | `initialize_vocabulary` |
| 3.2 | `get_pair_frequencies` |
| 3.3 + 3.5 | `train` |
| 3.4 | `merge_pair` |
| 3.6 | `encode_word` — then write `encode` (split the sentence, encode each word) and `decode` |

**Order of work (important):** run `pytest tests/test_bpe.py -x`. The tests are arranged **in the same order as the algorithm**, so the first failure is always the next function to write. Don't jump around.

`train` also has to **store** things the tests look for (the docstring lists them): `merges`, `merge_frequencies`, `merge_ranks`, `word_freqs`, `token_to_id`, `id_to_token`. Write them *after* the loop, one by one, running the tests after each.

### 7.6 Typical beginner mistakes in BPE

* Forgetting the **frequency** in step 3.2 (counting a pair once per word).
* Using a **list** as a dict key (`TypeError`). Convert with `tuple(...)`.
* Changing the table **while looping over it** — build a *new* dict instead.
* Merging from the wrong direction for overlapping pairs (`a a a`).
* Forgetting `</w>`.
* Forgetting to start `train` from a clean state.
* Testing in the notebook only: the tests in `pytest` check many more cases.

### 7.7 The experiments (BPE-10 / 30 / 50)

You train three tokenizers with different numbers of merges and look at their merge rules and the tokenization of known and unknown words. **Predict first:** after how many merges is there nothing left to merge on this tiny corpus? Then look at your table and explain what you see — in particular for BPE-30 and BPE-50.

### 7.8 The theory questions (C.5)

Don't answer from memory of a textbook. For each question pick **an example from your own runs** (a merge, a word, a number) and build the answer around it.

---

## 8. Part D — BPE on real data

**File:** `notebooks/04_bpe_real_data.ipynb` (3 exercises, no new `src/` code).

### 8.1 Before running

* The notebook trains BPE on **Shakespeare** (about one million characters). With 1000 merges it can take **one to three minutes** — that is normal; your toy corpus was instant.
* While you develop, set `DEV = True` (in the cell near the top): it uses a small part of the data so you get fast feedback. **Set it back to `False`** for the final run that produces your numbers.
* If it is *very* slow (more than ~10 minutes), stop the kernel (**Kernel → Interrupt**) and ask your teacher: it usually means a step in `train` is doing unnecessary work inside the loop.

### 8.2 Train versus test

The text is split into **train** lines (the tokenizer learns from them) and **test** lines (held-out, only for measuring). Why? A tokenizer always looks good on the text it learned from. Measuring on text it has *never seen* tells you how it behaves in real life.

### 8.3 The three exercises

* **4.1 (fertility and characters per token):** two ratios. *Fertility* = (tokens) / (words); *characters per token* = (characters) / (tokens). Both use **totals** over all lines. Use `sum(... for line in sample)` (§2.7) and `line.split()` for the words.
* **4.2 (unseen words):** build a **set** of the training words, then count the test words not in it. (A set is fast for `in`; a list would take forever.)
* **4.3 (token lengths):** `Counter(len(token) for token in vocabulary)` then a bar chart. The check explains how to treat the `</w>` marker.

### 8.4 What to look at in the results

* The **table** (vocabulary size, tokens per word, characters per token) and the **curves**: does adding merges help the same amount each time?
* The **first merges** (what kind of pairs are learned first?) and merges number 500–515.
* The **longest tokens**: do they look like language, or like something specific to this particular text? (Look at the first line of a play…)
* **Unseen words:** how does BPE cut a word it has never seen?

Write a sentence about each in the notebook, with the numbers.

---

## 9. Part E — WordPiece and Unigram (written)

No code. You write about one page, in your own words.

How to do it as a beginner:

1. First make sure you can explain BPE aloud to a friend (you can, after Part C).
2. Read the papers' *introductions and the algorithm sections*, not the whole papers. Take notes with these four questions for **each** method: *Where does the starting vocabulary come from? How does the vocabulary grow or shrink? What decides which pieces are kept or merged? How is a new word cut into pieces?*
3. Fill the comparison table with short, precise statements (one line per cell).
4. Cite what you read (authors, year, title). **Don't copy sentences**; explain them with your own words.
5. If you do not understand a formula, write down *what it measures* in words; that is enough.

---

## 10. Part F — Embeddings

**Files:** `notebooks/05_embeddings.ipynb`, `src/embeddings.py`, tests: `pytest tests/test_embeddings.py`.

### 10.1 The idea

A tokenizer produces **ids** (integers). The model needs **vectors** (lists of numbers). An *embedding table* has **one row per id**; each row is the vector for that id. "Embedding a token" means *reading one row*.

```
id 0 → [ 0.01, -0.02,  0.03, …]
id 1 → [ 0.05,  0.00, -0.01, …]
id 2 → [-0.03,  0.02,  0.04, …]
```
If there are V ids and each vector has d numbers, the table has V rows and d columns: shape `(V, d)`, with `V × d` numbers in total. Those numbers are the **parameters** that the model learns.

### 10.2 Practise with different numbers (so you can solve yours)

```python
import numpy as np
table = np.array([[10, 11], [20, 21], [30, 31]])    # V = 3, d = 2   -> shape (3, 2), 6 numbers
ids = [2, 0, 2]
print(table[ids])          # rows 2, 0, 2   -> shape (3, 2): one row per id
print(np.eye(3)[[2, 0]])   # one-hot rows for ids 2 and 0
print(np.array([1, 0, 0]) @ table)   # [10 11]  = row 0
```
The **rule for shapes**: the output has the shape of the id array **plus** `d` at the end. 3 ids → `(3, 2)`. A batch of 2 sentences with 5 ids each has ids of shape `(2, 5)` → output `(2, 5, d)`.

The last line shows the surprising fact behind exercise 5.2: multiplying a **one-hot vector** by the table gives exactly one row. So "lookup" and "matrix product" are the same thing; lookup is just much cheaper.

### 10.3 The exercises

* **5.1** — create a table with `rng.normal(...)`, then look up rows with `E[ids]`.
* **5.2** — build one-hot vectors with `np.eye(V)[ids]` and multiply by `E` using `@`.
* **5.3** — batches and a parameter count.

The notebook then asks the questions for V = 10 000 and d = 256: work them out **on paper first** using the rules above, *then* run the code to check.

### 10.4 Packaging

* `__call__` must also **reject ids that are out of range** (negative or ≥ V) with an `IndexError`. Note: NumPy accepts `-1` (it means "last row") — the tests want an error instead, so you need an explicit check.
* `one_hot` and `lookup_via_matmul` reuse what you learned.

---

## 11. The report

Your report is a short **argument** supported by your numbers, not a diary.

Sentence starters that help:

* "We observed that … (table X shows …)."
* "This happens because … (in BPE, … )."
* "A limitation of this experiment is … (for example, a single corpus)."
* "We chose NFC rather than … because …."

Checklist:

- [ ] every number comes from your notebooks (`results/`);
- [ ] decisions explained (cleaning order, tie rule, unknown characters…);
- [ ] figures have titles, axis labels and a caption;
- [ ] you cite your sources (papers, documentation, any AI tool you used for explanations);
- [ ] you can explain every figure and every function in a short oral check.

---

## 12. Troubleshooting FAQ

**"My terminal says `pytest: command not found`."**
Your virtual environment is not active. Activate it (§1.3) and run `pip install -r requirements.txt` again.

**"`ModuleNotFoundError: No module named 'src'`."**
Run `pytest` from the project's main folder (where `README.md` is). In notebooks, the first cell sets the path for you: run it first.

**"I changed `src/bpe.py` but the notebook behaves as before."**
The notebook still has the old version in memory. **Kernel → Restart**, then run the cells again.

**"The ✅ cell fails but my result looks right."**
Print your result and the expected one next to each other. Typical causes: a missing `return`, a list instead of a tuple, the wrong order of items, a forgotten marker `</w>`.

**"I filled all blanks and I get `SyntaxError`."**
Compare brackets and quotes: every `(` needs a `)`, every opening quote a closing quote. Check that you did not leave `____` somewhere.

**"All the tests in a file fail with the same error."**
Read the first failure only (`pytest -x`). One missing function often breaks many tests.

**"Everything is red and I don't know where to start."**
Go back to §3: pick the *first* failing test, read its name, reproduce in a scratch cell, fix, repeat.

**"Is it okay to look for solutions online or ask an AI to write the code?"**
Learning from explanations is fine (and you must cite them). **Pasting code you don't understand is not**: you will be asked to explain your code, and the check cells and tests cannot save you in an oral discussion. A good use of an AI assistant: ask it to explain an error message or a Python concept, *not* to solve the exercise.

**"My code is slow."**
First, is it a notebook with `DEV = False` on the real corpus? Training 1000 merges takes minutes. Second, are you using a **list** for membership tests (`x in a_list`) where a **set** would be instant? Third, are you recomputing something inside a loop that could be computed once before it?

---

## 13. Glossary

| Word | Meaning |
|---|---|
| **token** | one piece of text produced by a tokenizer (character, word or subword) |
| **tokenizer** | the program that cuts text into tokens and maps them to ids |
| **vocabulary** | the table token ↔ id |
| **id** | the integer number of a token in the vocabulary |
| **special token** | control symbol such as `[PAD]`, `[UNK]`, `[BOS]`, `[EOS]` |
| **OOV** | "out of vocabulary": an input unit the tokenizer has no entry for |
| **Unicode** | the standard that gives every character a number (code point) |
| **UTF-8** | a way of writing Unicode numbers as bytes |
| **normalization** | choosing one standard representation of equivalent text |
| **NFC / NFD** | two Unicode normal forms (composed / decomposed) |
| **corpus** | a collection of texts |
| **duplicate** | the same document appearing more than once |
| **BPE** | Byte Pair Encoding: a subword method that repeatedly merges the most frequent neighbouring pair |
| **merge rule** | one decision "glue symbol A and symbol B" |
| **end-of-word marker (`</w>`)** | extra symbol marking where a word ends |
| **fertility** | average number of tokens per word |
| **embedding** | a vector (list of numbers) assigned to a token id |
| **embedding matrix** | the table with one embedding per id |
| **parameter** | a number the model learns (all entries of the embedding matrix are parameters) |
| **unit test** | a small automatic check of one behaviour of your code |
| **assertion** | a statement the program checks (`assert x == 3`); it fails loudly if false |
| **kernel** | the Python process behind a notebook |

---

## 14. Plan and final checklist

### A realistic plan for a beginner (about 14 hours)

| Day | Work | Done when |
|---|---|---|
| 1 | Setup (§1), Python survival kit (§2), run `pytest` once | you can run a notebook cell and a test |
| 2 | Part A: exercises 1.1–1.5, then `src/preprocessing.py` | `test_preprocessing.py` green |
| 3 | Part B: exercises 2.1–2.4, then the two tokenizers | `test_tokenizers.py` green |
| 4–5 | Part C: by-hand merges, exercises 3.1–3.6 | all 3.x ✅ |
| 5–6 | `src/bpe.py` with `pytest -x` | `test_bpe.py` green |
| 7 | Notebooks 03 and 04 experiments | tables + figures saved |
| 8 | Part F (embeddings) + Part E (reading and writing) | `test_embeddings.py` green, page written |
| 9 | Report, final re-run of every notebook, checklist | submitted |

### Final checklist

**Code**
- [ ] All 21 🧩 exercises completed: no `____` left, all ✅ checks pass.
- [ ] `pytest` is fully green.
- [ ] You did not edit the provided tests.

**Experiments**
- [ ] Notebooks 01–05 run from top to bottom (**Kernel → Restart & Run All**) without errors.
- [ ] `DEV = False` in notebook 04 for the final run.
- [ ] Tables and figures saved in `results/`.

**Understanding** — you can explain, in your own words and without notes:
- [ ] why a careful cleaning pipeline is safer than an aggressive one (with an example);
- [ ] what a vocabulary is and why `[UNK]` exists;
- [ ] the six steps of BPE and the tie rule;
- [ ] why the order of merge rules matters;
- [ ] what an embedding lookup does.

**Report**
- [ ] Numbers come from your runs, sources cited, decisions explained.

---

*You don't need to be fast or brilliant. You need to be patient, to look at your data, and to take small steps. Every programmer you admire started exactly where you are now.*