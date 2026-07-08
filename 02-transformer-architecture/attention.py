import numpy as np
import math

vocab_size = 2256
embed_dim = 64
max_seq_len = 128

embedding_matrix = np.random.randn(vocab_size, embed_dim)
print(f"embedding lookup shape: ${embedding_matrix.shape}")

position_matrix = np.random.randn(max_seq_len, embed_dim)

token_ids = [348, 284, 101, 101, 112] 
seq_len = len(token_ids)

position_embeddings = position_matrix[:seq_len]
token_embeddings = embedding_matrix[token_ids]

x = token_embeddings + position_embeddings

print(f"x shape: {x.shape}")

Q_W = np.random.randn(embed_dim, embed_dim) * 0.1
K_W = np.random.randn(embed_dim,embed_dim)  * 0.1
V_W = np.random.randn(embed_dim,embed_dim)  * 0.1

Q = x @ Q_W
K = x @ K_W
V = x @ V_W

print(f"Q shape: {Q.shape}")
print(f"K shape: {K.shape}") 
print(f"V shape: {V.shape}")

scores = Q @ K.T
scores = scores / math.sqrt(embed_dim)
exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

output = attention_weights @ V

print(f"Output shape: {output.shape}")
print(f"Attention weights (who pays attention to whom):")
print(attention_weights)
print(f"Row sums: {attention_weights.sum(axis=1)}")
print(f"Input shape == Output shape: {x.shape == output.shape}")