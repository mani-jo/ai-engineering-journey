import numpy as np
from distance import cosine_similarity


vectors = {}

def add(id:str, vector):
    vectors[id] = vector

def search(query_vector, top_k = 2):
    results = []

    for id, vector in vectors.items():
        similarity = cosine_similarity(query_vector, vector)
        results.append((id, similarity))
    
    results.sort(key=lambda x: x[1], reverse=True)

    return results[:top_k]

if __name__ == "__main__":

    # Let's add some mock documents (pretend these vectors came from a model)
    add("doc_1_apple", np.array([0.9, 0.1, 0.1]))
    add("doc_2_banana", np.array([0.8, 0.2, 0.1]))
    add("doc_3_car", np.array([0.1, 0.8, 0.9]))
    add("doc_4_truck", np.array([0.1, 0.9, 0.8]))
    
    # A user searches for "fruit" (and the model gives us this vector)
    query_fruit = np.array([0.85, 0.15, 0.1])
    
    print("Searching for 'fruit'...")
    top_matches = search(query_fruit, top_k=2)
    
    for doc_id, score in top_matches:
        print(f"Match: {doc_id} (Score: {score:.4f})")
