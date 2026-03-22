documents = [
    "Cat chases boy",
    "Boy chases cat",
    "Cat eats fish",
    "Boy eats candy",
]


def preprocess(doc):
    return doc.lower().split()


def vectorize(doc, vocab):
    vector = [0] * len(vocab)
    words = preprocess(doc)
    for word in words:
        index = vocab[word]
        vector[index] += 1
    return vector


vocab = {}

for doc in documents:
    words = preprocess(doc)
    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab)


vectors = [vectorize(doc, vocab) for doc in documents]

for i, vec in enumerate(vectors):
    print(f"D{i+1}: {vec}")
