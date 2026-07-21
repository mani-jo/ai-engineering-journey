from sentence_transformers import SentenceTransformer
from brute_force import add, search

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = {
    "doc_1": "Our return policy allows full refunds within 30 days of purchase.",
    "doc_2": "The company cafeteria serves pizza on Fridays.",
    "doc_3": "To reset your password, click the link sent to your email.",
    "doc_4": "Employees get 20 days of paid time off per year."
}

print(f"Embedding documents and adding to my vector db")

for doc_id, doc in documents.items():
    vector = model.encode(doc)
    add(doc_id, vector=vector)

user_query = str(input("Ask your question: "))

query_vector = model.encode(user_query)

print(f"Performing brute force to find relevant answers")

top_matches = search(query_vector, 1)

for doc_id, score in top_matches:
    document = documents[doc_id]
    print(f"Match: {doc_id} (Score: {score:.4f})")
    print(f"Wording: {document}")
    

