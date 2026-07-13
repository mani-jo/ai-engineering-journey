from multi_head_attention import multi_head_attention
from feed_forward import feed_forward, layer_norm

def transformer_block(x, num_heads, Q_W, K_W, V_W, O_W, W1, b1, W2, b2):

    output = multi_head_attention(x, num_heads, Q_W, K_W, V_W, O_W)
    output = layer_norm(x + output)

    ff_output = feed_forward(output, W1, b1, W2, b2)
    ff_output = layer_norm(output + ff_output)

    return ff_output