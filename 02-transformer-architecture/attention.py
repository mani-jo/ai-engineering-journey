import numpy as np
from multi_head_attention import multi_head_attention

vocab_size = 2256
embed_dim = 64
head_dim = 16
num_heads = embed_dim // head_dim
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

Q_W = [np.random.randn(embed_dim, head_dim) * 0.1 for _ in range(num_heads)]
K_W = [np.random.randn(embed_dim, head_dim) * 0.1 for _ in range(num_heads)]
V_W = [np.random.randn(embed_dim, head_dim) * 0.1 for _ in range(num_heads)]

O_W = np.random.randn(embed_dim, embed_dim) * 0.1

output = multi_head_attention(x, num_heads, Q_W, K_W, V_W, O_W)

print(f"Output shape: {output.shape}, X shape: ${x.shape}")
print(f"Input shape == Output shape: {x.shape == output.shape}")