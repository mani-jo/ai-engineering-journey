import json
from tokenizer import encode, decode

def load_merges():
    with open("merges.json", 'r') as file:
        merges = json.load(file)
    
    return [tuple(m) for m in merges]

print(f"Loading tokenizer merges rules")
merges = load_merges()
print(f"Rules loaded")

user_input = str(input("enter text to tokenize: "))
tokens = encode(user_input, merges)

print(f"Tokens consumed: {len(tokens)}")
print(f"Tokens: {tokens}")

print(f"Original user input: {decode(tokens)}")


