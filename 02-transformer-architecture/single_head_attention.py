import numpy as np
import math

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def single_head_attention(input, Q_W, K_W, V_W):
    Q = input @ Q_W
    K = input @ K_W
    V = input @ V_W

    scores = Q @ K.T
    scores = scores / math.sqrt(Q.shape[1])

    weights = softmax(scores)
    output = weights @ V

    return output, weights

