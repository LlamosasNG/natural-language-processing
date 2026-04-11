# 📝 Mock Exam — NLP Fundamentals (RESUELTO)

**Fecha:** miércoles, 18 de marzo de 2026

---

## Pregunta 1 — Definiciones Fundamentales

**Instrucción:** Define los siguientes términos.

| Término        | Definición                                                                                                                                                                                                                                                                                                                                                                                          |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Corpus**     | Colección grande y estructurada de textos (escritos o hablados) utilizada para análisis lingüístico o entrenamiento de modelos. Proviene directamente del latín, donde significa “cuerpo”. En la erudición clásica y medieval, corpus se usaba metafóricamente para significar: un conjunto de datos, una colección acotada de escritos, un canon perteneciente a un autor, tradición o disciplina. |
| **Vocabulary** | Conjunto de tipos de palabras únicas conocidas por un modelo o un conjunto de datos.                                                                                                                                                                                                                                                                                                                |
| **Token**      | Unidades individuales que se generan mediante la tokenización (generalmente palabras, signos de puntuación o símbolos).                                                                                                                                                                                                                                                                             |
| **Type**       | Elemento de vocabulario distinto, independientemente de la frecuencia con la que aparezca.                                                                                                                                                                                                                                                                                                          |
| **Alphabet**   | Conjunto finito de símbolos no vacíos a partir del cual se construyen cadenas.                                                                                                                                                                                                                                                                                                                      |
| **String**     | Secuencia finita de símbolos de un alfabeto específico.                                                                                                                                                                                                                                                                                                                                             |

---

## Pregunta 2 — Tokenización y Lematización

Dada la siguiente oración:

> _"The cats are chasing the mice in the gardens."_

1. Lista los **tokens**.
2. Lista los **lemas** de todos los verbos y sustantivos de la oración.
3. Explique un **caso excepcional** en el que un tokenizador simple con espacios en blanco fallaría.

### 1. Lista de tokens

```
["The", "cats", "are", "chasing", "the", "mice", "in", "the", "gardens", "."]
```

### 2. Lemas de verbos y sustantivos

| Token   | Categoría  | Lema   |
| ------- | ---------- | ------ |
| cats    | Sustantivo | cat    |
| are     | Verbo      | be     |
| chasing | Verbo      | chase  |
| mice    | Sustantivo | mouse  |
| gardens | Sustantivo | garden |

### 3. Caso problemático para un tokenizador

## `Don't`, `Week-end`

Un tokenizador simple **fallaría** con contracciones como **`Don't`**. Un tokenizador por espacios lo trataría como un solo token `"Don't"`, cuando en realidad debería separarse en dos tokens: `"Do"` y `"n't"` (o `"Don"` y `"'t"`). Esto es importante porque `"not"` cambia totalmente el significado de la oración y debe tratarse como unidad separada.

Otro ejemplo es **`Week-end`**: un tokenizador por espacios lo trataría como un solo token, cuando podría necesitar separarse en `"Week"` y `"end"`, o mantenerse unido según el contexto.

---

## Pregunta 3 — Lenguajes Formales y Cadenas

Sea el alfabeto **Σ = {a, b, c}**.

1. ¿Cuántas cadenas de **longitud 3** se pueden formar sobre Σ?
2. Lista las **10 cadenas más pequeñas** en Σ\*.

### 1. Cadenas de longitud 3

Se pueden formar **27 strings** de longitud 3 ya que para cada posición del string tenemos 3 opciones (a, b o c). Como hay 3 posiciones, multiplicamos: 3 × 3 × 3 = 27.

### 2. Las 10 cadenas más pequeñas en Σ\*

| #   | String           | Longitud |
| --- | ---------------- | -------- |
| 1   | ε (cadena vacía) | 0        |
| 2   | a                | 1        |
| 3   | b                | 1        |
| 4   | c                | 1        |
| 5   | aa               | 2        |
| 6   | ab               | 2        |
| 7   | ac               | 2        |
| 8   | ba               | 2        |
| 9   | bb               | 2        |
| 10  | bc               | 2        |

---

## Pregunta 4 — Expresiones Regulares

Escribe **expresiones regulares compatibles con Python** (módulo `re`) para encontrar:

1. Todos los **hashtags válidos** en un tweet
   - Ejemplos: `#NLP2024`, `#machine_learning`

2. **Fechas** en formato `DD/MM/YYYY` o `DD-MM-YYYY`

### 1. Hashtags válidos en un tweet

```python
pattern1 = r"#\w+"
pattern2 = r"#[A-Za-z_]\w*"
```

### 2. Fechas en formato DD/MM/YYYY o DD-MM-YYYY

```python
pattern1 = r"\b(0[1-9]|[12][0-9]|3[01])([-/])(0[1-9]|1[0-2])\2(\d{4})\b"
pattern2 = r"\b\d{2}[/-]\d{2}[/-]\d{4}\b"
```

---

## Pregunta 5 — Modelos de Lenguaje N-gramas

Considera el siguiente toy corpus:

- **D1:** "I love NLP."
- **D2:** "I do not love NLP."

1. Construye el **vocabulario de Bigramas (2-gramas)** para este corpus.
2. Representa **D2** como un vector usando el vocabulario de bigramas.

### Primero tokenizamos (sin puntuación):

- D1: `[I, love, NLP]`
- D2: `[I, do, not, love, NLP]`

1. Bigramas de D1: `(I love)`, `(love NLP)`
2. Bigramas de D2: `(I do)`, `(do not)`, `(not love)`, `(love NLP)`

**_Vocabulario de Bigramas `(I love)`, `(love NLP)`, `(I do)`, `(do not)`, `(not love)`_**

### 2. Vector de D2 usando el vocabulario de bigramas

Contamos la frecuencia de cada bigrama del vocabulario en D2:

| Bigrama    | ¿Aparece en D2? |
| ---------- | --------------- |
| (I love)   | 0               |
| (love NLP) | 1               |
| (I do)     | 1               |
| (do not)   | 1               |
| (not love) | 1               |

**Vector de D2:** `[0, 1, 1, 1, 1]`

---

## Pregunta 6 — TF-IDF

Tienes un corpus de **1,000 documentos**.

- La palabra **"transformer"** aparece en **10 documentos**.
- El **Documento A** tiene **100 palabras en total**, y "transformer" aparece **5 veces**.

1. Calcula la **Frecuencia de Término (TF)** de "transformer" en el Documento A.
2. Calcula la **Frecuencia Inversa de Documento (IDF)** usando el logaritmo en base 10.
3. Calcula el **puntaje TF-IDF final** de "transformer" en el Documento A.

### 1. Term Frequency (TF)

$$TF(\text{transformer}, A) = 5$$

### 2. Inverse Document Frequency (IDF)

$$IDF(\text{transformer}) = \log_{10}\left(\frac{N}{df(t)}\right) = \log_{10}\left(\frac{1000}{10}\right) = \log_{10}(100) = 2$$

### 3. TF-IDF Score

$$TF-IDF(\text{transformer}, A) = TF \times IDF = 5 \times 2 = 10$$

---

## Pregunta 7 — TF-IDF en Python (Colab)

Implementa lo siguiente en Python:

**a)** Carga un conjunto de datos de texto pequeño

- Por ejemplo, el dataset **20 Newsgroups** de scikit-learn, o una muestra descargada de internet.

**b)** Preprocesa el conjunto de datos:

- Convierte todo el texto a minúsculas
- Elimina la puntuación
- Tokeniza

**c)** Construye una **matriz TF-IDF** usando `scikit-learn`.

**d)** Imprime:

- El **tamaño del vocabulario**
- El **vector TF-IDF del primer documento** (solo las primeras 10 entradas)

```python

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import re

categories = ['rec.sport.baseball']
newsgroups = fetch_20newsgroups(subset='train', categories=categories)
documents = newsgroups.data
print(f"Número de documentos cargados: {len(documents)}")

def preprocess(text):
   text = text.lower()
   text = re.sub(r'[^\w\s]', '', text)
   tokens = text.split()
   return ' '.join(tokens)

documents_clean = [preprocess(doc) for doc in documents]
print(f"Ejemplo de documento preprocesado:\n{documents_clean[0][:200]}...")

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents_clean)
tfidf_vocabulary = vectorizer.get_feature_names_out()
print(f"\nVocabulario TF-IDF:", tfidf_vocabulary)

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vocabulary)
print("\nTF-IDF Matrix (dense format):\n")
print(tfidf_df)
```

---

## Pregunta 8 — Embeddings de Palabras con Gensim

Usando `gensim`, realiza lo siguiente:

**a)** Carga o crea **embeddings de palabras**

- Usa Word2Vec de `gensim` entrenado sobre un texto pequeño.

**b)** Muestra las **5 palabras más similares** a la palabra `"language"`.

**c)** Calcula la **similitud coseno** entre:

- `"cat"` y `"dog"`
- `"cat"` y `"car"`

```python
!pip install gensim

import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
nltk.download('punkt')
nltk.download('punkt_tab')

corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "A dog is a man's best friend. They are loyal and playful.",
    "NLP is a fascinating field. It helps computers understand human language.",
    "Machine learning algorithms are at the core of many AI applications.",
    "The weather today is sunny and warm. Perfect for an outdoor adventure.",
    "Data science combines statistics, computer science, and domain expertise.",
    "Natural Language Processing involves tasks like text classification and sentiment analysis.",
    "Python is a popular language for data analysis and machine learning."
]

print(f"Corpus created with {len(corpus)} documents.")
print("First document:", corpus[0])

sentences = [word_tokenize(doc.lower()) for doc in corpus]

print("First tokenized sentence:", sentences[0])

model = Word2Vec(min_count=1, vector_size=100, window=5, seed=42)

model.build_vocab(sentences)

model.train(sentences, total_examples=model.corpus_count, epochs=10)

print("\nWord2Vec model training complete.")
word_to_find_vector = 'dog'
if word_to_find_vector in model.wv:
    dog_vector = model.wv[word_to_find_vector]
    print(f"\nVector for '{word_to_find_vector}':\n{dog_vector[:10]}...")
else:
    print(f"\n'{word_to_find_vector}' not in vocabulary.")

word_for_similarity = 'cat'
if word_for_similarity in model.wv:
    similar_words = model.wv.most_similar(word_for_similarity, topn=5)
    print(f"\nWords most similar to '{word_for_similarity}':\n{similar_words}")
else:
    print(f"\n'{word_for_similarity}' not in vocabulary.")

```