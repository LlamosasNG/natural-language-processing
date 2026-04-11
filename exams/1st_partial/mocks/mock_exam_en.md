# 📝 Mock Exam — NLP Fundamentals
**Fecha:** miércoles, 18 de marzo de 2026

---

## Question 1 — Core Definitions

Define the following terms:

| Term | Definition |
|------|-----------|
| **Corpus** | |
| **Vocabulary** | |
| **Token** | |
| **Type** | |
| **Alphabet** | |
| **String** | |

---

## Question 2 — Tokenization & Lemmatization

Given the sentence:

> *"The cats are chasing the mice in the gardens."*

1. List the **tokens**.
2. List the **lemmas** for all verbs and nouns in the sentence.
3. Explain one **edge case** where a simple whitespace tokenizer would fail.

> 💡 **Hint — edge case examples:** `Don't`, `Week-end`

---

## Question 3 — Formal Languages & Strings

Let the alphabet **Σ = {a, b, c}**.

1. How many strings of **length 3** can be formed over Σ?
2. List the **smallest 10 strings** in Σ\* (lexicographic order).

---

## Question 4 — Regular Expressions

Write **Python-compatible Regular Expressions** (`re` module) to match:

1. All valid **hashtags** in a tweet
   - Examples: `#NLP2024`, `#machine_learning`

2. **Dates** in the format `DD/MM/YYYY` or `DD-MM-YYYY`

---

## Question 5 — N-gram Language Models

Consider the following toy corpus:

- **D1:** "I love NLP."
- **D2:** "I do not love NLP."

1. Construct a **Bigram (2-gram) vocabulary** for this corpus.
2. Represent **D2** as a vector using the Bigram vocabulary.

---

## Question 6 — TF-IDF

You have a corpus of **1,000 documents**.
- The word **"transformer"** appears in **10 documents**.
- **Document A** has **100 total words**, and "transformer" appears **5 times**.

### Tasks

1. Calculate the **Term Frequency (TF)** of "transformer" in Document A.
2. Calculate the **Inverse Document Frequency (IDF)** using the base-10 logarithm.
3. Calculate the final **TF-IDF score** for "transformer" in Document A.

### Formulas

**Term Frequency:**

$$TF(t, d) = \text{count of term } t \text{ in document } d$$

$$TF(\text{transformer}, A) = 5$$

**Inverse Document Frequency:**

$$IDF(t) = \log_{10}\left(\frac{N}{df(t)}\right)$$

- $N$ = Total number of documents
- $df(t)$ = Number of documents containing the term

**TF-IDF:**

$$TFIDF(t, d) = TF(t, d) \times IDF(t)$$

---

## Question 7 — TF-IDF in Python (Colab)

Implement the following in Python:

**a)** Load a small text dataset
  - e.g., scikit-learn's **20 Newsgroups**, or a small sample downloaded from the internet.

**b)** Preprocess the dataset:
  - Lowercase all text
  - Remove punctuation
  - Tokenize

**c)** Construct a **TF-IDF matrix** using `scikit-learn`.

**d)** Print:
  - The **size of the vocabulary**
  - The **TF-IDF vector of the first document** (first 10 entries only)

---

## Question 8 — Word Embeddings with Gensim

Using `gensim`, perform the following:

**a)** Load or create **word embeddings**
  - Use `gensim`'s Word2Vec trained on a small text.

**b)** Show the **5 most similar words** to the word `"language"`.

**c)** Compute the **cosine similarity** between:
  - `"cat"` and `"dog"`
  - `"cat"` and `"car"`