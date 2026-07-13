# gpt.py — The full GPT pipeline
import numpy as np
from transformer_block import transformer_block
from single_head_attention import softmax

# === CONFIG ===
vocab_size = 2256
embed_dim = 64
head_dim = 16
num_heads = embed_dim // head_dim
max_seq_len = 128
num_layers = 4        # GPT-2 uses 12, we use 4 to keep it fast
hidden_dim = embed_dim * 4


# === EMBEDDING MATRICES (shared) ===
embedding_matrix = np.random.randn(vocab_size, embed_dim)
position_matrix = np.random.randn(max_seq_len, embed_dim)

# === CREATE WEIGHTS FOR EACH LAYER ===
# Each layer gets its own set of weights
layers = []
for i in range(num_layers):
    layer_weights = {
        'W_Qs': [np.random.randn(embed_dim, head_dim) * 0.1 for _ in range(num_heads)],
        'W_Ks': [np.random.randn(embed_dim, head_dim) * 0.1 for _ in range(num_heads)],
        'W_Vs': [np.random.randn(embed_dim, head_dim) * 0.1 for _ in range(num_heads)],
        'W_O':   np.random.randn(embed_dim, embed_dim) * 0.1,
        'W1':    np.random.randn(embed_dim, hidden_dim) * 0.1,
        'b1':    np.zeros(hidden_dim),
        'W2':    np.random.randn(hidden_dim, embed_dim) * 0.1,
        'b2':    np.zeros(embed_dim),
    }
    layers.append(layer_weights)

# === FORWARD PASS ===
token_ids = [348, 284, 101, 101, 112]
seq_len = len(token_ids)

# Step 1: Embedding + Position
x = embedding_matrix[token_ids] + position_matrix[:seq_len]
print(f"Input shape: {x.shape}")



# Step 2: Pass through ALL transformer blocks
for i, layer_weights in enumerate(layers):
    x = transformer_block(
        x, num_heads,
        layer_weights['W_Qs'], layer_weights['W_Ks'],
        layer_weights['W_Vs'], layer_weights['W_O'],
        layer_weights['W1'],   layer_weights['b1'],
        layer_weights['W2'],   layer_weights['b2']
    )
    print(f"After layer {i+1}: {x.shape}")

print(f"Final output shape: {x.shape}")   # (5, 64) — same as input!

logits = x @ embedding_matrix.T
probs = softmax(logits)
predicted_id = np.argmax(probs[-1])
print(f"Predicted token ID: {predicted_id}")