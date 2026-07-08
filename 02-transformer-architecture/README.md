# Day 2: Self-Attention (The Transformer Engine)

## What is this?
A pure Python + NumPy implementation of the Self-Attention mechanism — the core algorithm inside every LLM (GPT-2, GPT-4, Claude, Llama). No PyTorch, no TensorFlow.

## What it does
Self-Attention solves one problem: **the same word means different things in different contexts.**

The word "bank" has the same token ID whether you mean a financial bank or a river bank. Self-Attention lets "bank" look at surrounding words ("deposit" vs "river") and transform its internal representation to match the correct meaning.

## How it works
Every token generates three vectors:
- **Query (Q):** What am I looking for?
- **Key (K):** What do I offer?
- **Value (V):** What is my actual content?

Each token's Query is scored against every other token's Key (dot product). High scores mean high relevance. The scores are normalized (softmax) into percentages, and each token absorbs a weighted sum of all Values.

```
Input:  [seq_len × embed_dim]   e.g. (3, 64)
Output: [seq_len × embed_dim]   e.g. (3, 64)  ← same shape, richer meaning
```

## Key concepts implemented
- **Embedding Matrix lookup** — converting token IDs to 64-dimensional vectors
- **Positional Encoding** — teaching the model word order
- **Scaled Dot-Product Attention** — Q × K^T / sqrt(d), then softmax, then × V
- **Weight initialization** — why `* 0.1` matters (prevents spiky attention)

## Why O(n²)?
Every token scores against every other token. For `n` tokens, that's `n × n` score calculations. This is the fundamental bottleneck of all Transformer models.

## Run it locally
```bash
python3 attention.py
```
