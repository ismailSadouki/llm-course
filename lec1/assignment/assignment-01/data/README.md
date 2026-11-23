# Data used in Assignment 01

| Folder | File | What it is | Use it for |
|---|---|---|---|
| `toy/` | `toy_corpus.txt` | 9 words: `low`x3 `lower`x2 `newer`x2 `newest`x2 | Part C: trace BPE by hand and verify your code |
| `toy/` | `toy_corpus_classic.txt` | The classic example from the BPE literature (`low`x5 `lower`x2 `newest`x6 `widest`x3) | Extra sanity check (used by a test) |
| `raw/` | `messy_documents.jsonl` | 24 hand-made noisy records: HTML, `null`, empty strings, NFD/NFC, French, Arabic, Darija, exact duplicates | Part A: preprocessing |
| `real/` | `tinyshakespeare.txt` | ~1.1 MB of Shakespeare plays (public-domain text, ~40k lines), as distributed with Karpathy's `char-rnn` repository | Parts B and D: realistic experiments |

Notes on the real dataset
* It is **English, archaic, and uppercase speaker names are frequent** ("MENENIUS:"). That is a property of this corpus: watch for it when you inspect the learned merges.
* Always compute metrics on **held-out lines** (`split_lines` in `src/evaluation.py`, seed 0, 20 % test) -- never on the lines you trained on.
* If the file is missing: `python scripts/download_data.py`.
