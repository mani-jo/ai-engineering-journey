import json
from tokenizer import train_bpe, build_vocab

with open("input.txt", "r") as file:
    text = file.read()

text = text[:20000]

final_words, merges = train_bpe(text=text, num_merges=2000)
build_vocab(merges)


with open("merges.json", "w") as file:
    json.dump(merges,file)