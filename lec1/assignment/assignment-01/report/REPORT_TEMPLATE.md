# Assignment 01 — Technical report (6–8 pages)

Names: …            Date: …            Repository: …

## 1. Introduction (≤ ½ page)
What pipeline did you study (raw text → cleaning → normalization → tokenization → ids → embeddings) and what question did you answer?

## 2. Preprocessing (Part A)
* Your pipeline and the order of its steps (with justification).
* Table from notebook 01: original / aggressive / conservative for the 6 sentences.
* Which information is lost by each aggressive step? Which sentences collided?
* Policy decisions the tests do not cover (newlines, zero-width characters, NFC vs NFKC).

## 3. Character and word tokenizers (Part B)
* Convention for the word tokenizer (punctuation, apostrophes, case, whitespace).
* Table: vocabulary size, tokens on the test set, OOV rate (min_freq sweep) + the figure.
* Discussion: advantages / limitations of each.

## 4. BPE from scratch (Part C)
* Algorithm in your own words, with the toy-corpus trace of the first 3 merges (frequencies included) and the tie-breaking rule.
* Table BPE-10 / BPE-30 / BPE-50: merges learned, vocabulary size, tokens per word, unseen words.
* Why BPE-30 = BPE-50 here; answers to C.5.
* Design decisions: data structures, complexity of train / encode_word, what you do with unseen characters.

## 5. BPE on real data (Part D)
* Setup: corpus, preprocessing, split, merge budgets.
* Results table + figures (tokens/word and characters/token vs merges, token-length histogram).
* What the first merges look like, what unseen words look like, comparison with char / word tokenizers.
* Limitations (whitespace-only pre-tokenization, corpus artifacts such as speaker names, a single small corpus).

## 6. WordPiece and Unigram (Part E) (≈ 1 page)
Answers to the questions + the comparison table. Cite sources.

## 7. Embeddings (Part F)
Shapes, parameter counts, lookup vs one-hot matmul, link to vocabulary size.

## 8. Limitations and conclusion

## References
Use a consistent citation style. Every external claim needs a source.

## Appendix — reproducibility
Python version, how to run the tests, how to regenerate every figure/table, random seeds used.
