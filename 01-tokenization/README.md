# Day 1: Tokenization (Byte Pair Encoding)

## What is this?
A pure Python implementation of the Byte Pair Encoding (BPE) algorithm used by large language models like GPT-2 and GPT-4. Built from scratch without using `tiktoken` or HuggingFace.

## Why build this?
To move from Layer 2 mastery (using APIs) to Layer 4 mastery (building the black box). Building the tokenizer from scratch reveals:
- Why attention is $O(n^2)$ and why shorter token sequences save money.
- Why LLMs struggle with foreign languages and code if not trained on them.
- How an LLM can perfectly tokenize a word it has never seen before (like `"mug"`) without throwing an `<UNK>` error, by mathematically falling back to known subwords.

## The Architecture
This project is professionally structured exactly like a production NLP pipeline:

1. **`tokenizer.py`**: The core math/engine library containing the BPE algorithms.
2. **`train.py`**: Reads a large dataset (e.g., Tiny Shakespeare), extracts the subword merge rules, and saves them to `merges.json`.
3. **`inference.py`**: The "API" layer. Instantly loads the frozen rules from `merges.json` and encodes user input without retraining.

## Example Output
When training on Shakespeare and encoding the unseen word "Amandeep is the new shakespear":
```text
Tokens: [['A', 'm', 'and', 'e', 'e', 'p'], ['is'], ['the'], ['ne', 'w'], ['sh', 'ak', 'espe', 'ar']]
```
*Notice how common English words ("is", "the") are heavily compressed, while unknown names ("Amandeep") fall back to subword characters without breaking the tokenizer!*

## Run it locally

First, train the tokenizer on a dataset to generate `merges.json`:
```bash
python3 train.py
```

Then, run the inference script to test it:
```bash
python3 inference.py
```

Run the automated test suite to prove correctness:
```bash
python3 test_bpe.py
```
