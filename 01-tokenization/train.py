import json
from tokenizer import train_bpe

with open("input.txt", "r") as file:
    text = file.read()


vocab, merges = train_bpe(text=text, num_merges=5000)

with open("merges.json", "w") as file:
    json.dump(merges,file)