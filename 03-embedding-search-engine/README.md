# Phase 3: Embedding Search Engine

This project is a from-scratch implementation of a Vector Database / Search Engine. Before using tools like `pgvector` or Pinecone, I am building the core algorithms to understand exactly how semantic search works under the hood.

## The Core Concept: Cosine Similarity

Traditional databases search for exact word matches (e.g., `WHERE content LIKE '%refund%'`). This fails when a user searches for "money back" because the exact characters don't match.

A **Vector Database** solves this using Embeddings and Cosine Similarity.

1. **Embeddings**: We use a trained model to convert sentences into arrays of numbers (vectors). These vectors capture the _meaning_ of the text.
2. **Cosine Similarity**: We calculate the angle between two vectors.
   - We ignore the length (magnitude) of the vectors, which means a short tweet and a long book about the exact same topic will still match perfectly.
   - If the angle is 0 (score of `1.0`), the vectors point in the exact same direction. They have the same meaning.
   - We search by comparing the user's query vector against every document vector and returning the ones with a score closest to `1.0`.

## Study Resources

I studied these two foundational resources to understand the math and theory before writing the code:

1. **[Pinecone — Vector Similarity Explained](https://www.pinecone.io/learn/vector-similarity/)**
   _Explains the difference between Dot Product, Euclidean Distance (L2), and Cosine Similarity with visual examples._

2. **[Jay Alammar — The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)**
   _Explains how models actually learn to place words with similar meanings close together in vector space (the distributional hypothesis)._
