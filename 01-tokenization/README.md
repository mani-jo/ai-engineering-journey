# Day 1: Tokenization (Byte Pair Encoding)

## What is this?
A pure Python implementation of the Byte Pair Encoding (BPE) algorithm used by large language models like GPT-2 and GPT-4. Built from scratch without using `tiktoken` or HuggingFace.

## Why build this?
To move from Layer 2 mastery (using APIs) to Layer 4 mastery (building the black box). Building the tokenizer from scratch reveals:
- Why attention is $O(n^2)$ and why shorter token sequences save money.
- Why LLMs struggle with foreign languages and code if not trained on them.
- How an LLM can perfectly tokenize a word it has never seen before (like `"mug"`) without throwing an `<UNK>` error, by mathematically falling back to known subwords.

## How it works
1. **Train**: Scans a text corpus, starts with character-level tokens, and iteratively merges the most frequent adjacent pairs (e.g., `'u'` + `'g'` → `'ug'`).
2. **Encode**: Takes unseen text, breaks it down into characters, and applies the learned merge rules in the exact order they were trained.
3. **Decode**: Converts token arrays back into human-readable strings.

## Run it locally
```bash
python3 bpe.py
```
