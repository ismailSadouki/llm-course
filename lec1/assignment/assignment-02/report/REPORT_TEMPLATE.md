# Assignment 02 — Technical report (8–10 pages)

Names: …     Date: …     Repository: …     Library versions (tokenizers / transformers): …

## 1. Introduction (≤ ½ page)
Question studied, what you built, summary of the main finding.

## 2. The Hugging Face ecosystem (Part A) (≤ 1 page)
Roles of `tokenizers`, `transformers`, `datasets`, `huggingface_hub`; `AutoTokenizer`; fast vs slow; what a tokenizer configuration stores.

## 3. Pretrained tokenizers (Parts B, C)
Table: model, class, algorithm, `vocab_size`, `len(tokenizer)`, special tokens. Encoding/decoding examples; vocabulary observations (`Ġ`, `▁`, `##`); decoding differences and their causes.

## 4. Normalization and pre-tokenization (Part D)
Table of the normalization experiment; pre-tokenizer comparison; consequences for the final tokens.

## 5. Training custom tokenizers (Part E)
Corpus (source, size, language distribution), settings, learned vocabulary excerpts, requested vs final vocabulary size, effect of `min_frequency`, BPE vs WordPiece on the same 20 sentences, reproducibility experiment.

## 6. Saving, reloading, testing (Part F)
What `tokenizer.json` contains; consistency check results; `PreTrainedTokenizerFast`; what your tests cover.

## 7. Multilingual benchmark (Part G)
Setup (tokenizers, data, metrics, how special tokens and unknowns are handled), full table, per-language tables, parity table, the four required figures, timing method.

## 8. Discussion
Answers to the analysis questions, backed by numbers. Why token count alone is not a quality measure.

## 9. Final challenge (1–2 pages)
Design of a 32 000-token multilingual tokenizer (see the handout).

## 10. Limitations and conclusion

## References

## Appendix — reproducibility
Versions, commands, seeds, how to regenerate every table and figure.
