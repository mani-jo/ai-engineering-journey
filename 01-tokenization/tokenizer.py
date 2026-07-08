import json

def get_word_frequencies(text:str):
    """
    Takes raw text, return dictionary of:
    word (tuple of characters) - how many times it appeared
    """

    freq = {}
    words = text.split(" ")

    for word in words:
        chars = tuple(word)
        if chars in freq:
            freq[chars] += 1
        else:
            freq[chars] = 1

    return freq

def get_pairs(words_freq):
    """
    count every adjacent pairs in all words
    """

    pairs = {}

    for word, count in words_freq.items():

        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])

            if pair in pairs:
                pairs[pair] += count
            else:
                pairs[pair] = count

    return pairs


def merge_pair(word_freq, pair):
    new_word_freq = {}

    for word, count in word_freq.items():
        new_word = []
        i = 0

        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                merged = word[i] + word[i + 1]
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        new_word_freq[tuple(new_word)] = count

    return new_word_freq


def train_bpe(text, num_merges):
    """
    Train BPE tokenizer on given text
    """

    word_freqs = get_word_frequencies(text)
    merges = []

    print(f"Starting with input text")
    print(f"Init words: {word_freqs}")
    print(f"num of merge, will perform {num_merges}")

    for i in range(num_merges):
        pairs = get_pairs(word_freqs)

        if not pairs:
            print(f"no more pairs left, stopped at {i + 1}")
            break

        bp = max(pairs, key=lambda p : pairs[p])
        best_count = pairs[bp]

        word_freqs = merge_pair(word_freqs, bp)
        merges.append(bp)

        new_token = bp[0] + bp[1]

        print(f"Merge : {i + 1} -> '{bp[0]}' + '{bp[1]}' appears {best_count} and merged '{new_token}' new token")
        print(f"new word freq: {word_freqs}")
        print()

    return word_freqs, merges


def encode(text, merges): 
    """
    Tokenize new text using merge rules
    """

    print(f"Given text: {text}")

    words = text.split()
    tokenized = []

    for word in words:
        chars = list(word)
        for pair in merges:
            new_tokens = []
            i = 0
            
            while i < len(chars):
                if i < len(chars) - 1 and chars[i] == pair[0] and chars[i + 1] == pair[1]:
                    merged = chars[i] + chars[i + 1]
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(chars[i])
                    i += 1
            chars = new_tokens
        tokenized.append(chars)
    
    return tokenized


def decode(tokens):
    """
    convert back to original text
    """

    words = []
    for token in tokens:
        word = ''.join(token)
        words.append(word)
    
    return " ".join(words)


def build_vocab(merges):
    """
    Build vocab from merges rules
    """

    vocab = {}

    for i in range(256):
        vocab[chr(i)] = i
    
    for pair in merges:
        merged_token = "".join(pair)
        vocab[merged_token] = len(vocab)
    
    with open("vocab.json", "w") as file:
        json.dump(vocab, file)

    return vocab