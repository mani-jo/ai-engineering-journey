import numpy as np

def feed_forward(x, W1, b1, W2, b2):
    """
    x shape:     (seq_len, embed_dim)      e.g. (5, 64)
    W1 shape:    (embed_dim, hidden_dim)   e.g. (64, 256)
    b1 shape:    (hidden_dim,)             e.g. (256,)
    W2 shape:    (hidden_dim, embed_dim)   e.g. (256, 64)
    b2 shape:    (embed_dim,)              e.g. (64,)
    output shape: (seq_len, embed_dim)     e.g. (5, 64) ← same as input!
    """

    hidden = x @ W1 + b1
    hidden = np.maximum(0, hidden)  # ReLU (Rectified linear unit) activation
    hidden = hidden @ W2 + b2

    return hidden

def layer_norm(x):
    """
    Normalize each token's vector.
    x shape: (seq_len, embed_dim) → same shape out
    """
    mean = x.mean(axis=-1, keepdims=True)       # average of each token's 64 numbers
    std = x.std(axis=-1, keepdims=True) + 1e-8  # spread of each token's 64 numbers
    return (x - mean) / std