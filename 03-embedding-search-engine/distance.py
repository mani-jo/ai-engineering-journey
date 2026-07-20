import numpy as np

def dot_product(n1, n2):
    return np.dot(n1, n2)

def cosine_similarity(a, b):
    dot = dot_product(a, b)

    a = np.linalg.norm(a)
    b = np.linalg.norm(b)

    return dot / (a * b)

print(cosine_similarity([0.3,0.5],[0.3,0.1]))