import numpy as np
from single_head_attention import single_head_attention

def multi_head_attention(input, num_heads, Q_W, K_W, V_W, O_W):
    """
    Multi-head attention mechanism.
    1. run single head attention
    2. combine output
    3. return final output
    """

    head_outputs = []
    
    for i in range(num_heads):
        output, weights = single_head_attention(input, Q_W[i], K_W[i], V_W[i])
        print(f"Head {i+1} output shape: {output.shape}")
        print(f"Head {i+1} attention weights shape: {weights.shape}")

        head_outputs.append(output)

    
    concatenated = np.concatenate(head_outputs, axis=-1)

    output = concatenated @ O_W

    return output
