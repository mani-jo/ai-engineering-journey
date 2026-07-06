import json
from tokenizer import encode, decode

def load_merges():
    with open("merges.json", 'r') as file:
        merges = json.load(file)
    
    return [tuple(m) for m in merges]

def load_vocab():
    with open("vocab.json", "r") as file:
        vocab = json.load(file)
    
    return vocab

print(f"Loading tokenizer merges rules and vocab")
merges = load_merges()
vocab = load_vocab()
print(f"Rules loaded")

while True:
    user_input = str(input("enter text to tokenize or -1 to exit: "))

    if(user_input == "-1"):
        break

    encoded = encode(user_input, merges)
    str_tokens = [t for word in encoded for t in word]
    token_consumed = len(str_tokens)

    integer_ids = [vocab[str_token] for str_token in str_tokens]

    print(f"Tokens consumed: {token_consumed}")
    print(f"String Tokens: {str_tokens}")
    print(f"Integer Tokens: {integer_ids}")

    print(f"Original user input: {decode(encoded)}")


