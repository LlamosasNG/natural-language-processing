# 📝 Mock Exam — NLP Fundamentals (RESUELTO)

**Fecha:** miércoles, 18 de marzo de 2026

---

## Pregunta 1 — Definiciones Fundamentales

**Instrucción:** Define los siguientes términos.

| Término        | Definición                                                                                                                                                                                   |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Corpus**     | Colección grande y estructurada de textos (escritos o hablados) utilizada para análisis lingüístico o entrenamiento de modelos de NLP. Ejemplo: todos los artículos de Wikipedia en español. |
| **Vocabulary** | Conjunto de **palabras únicas** (types) presentes en un corpus. Si un corpus tiene 1000 palabras pero solo 300 son distintas, el vocabulario tiene tamaño 300.                               |
| **Token**      | Cada ocurrencia individual de una palabra o símbolo en un texto. En "el gato y el perro", hay **5 tokens** (la palabra "el" cuenta dos veces).                                               |
| **Type**       | Cada palabra **única/distinta** en un texto, sin importar cuántas veces aparezca. En "el gato y el perro", hay **4 types**: {el, gato, y, perro}.                                            |
| **Alphabet**   | Conjunto finito de símbolos a partir del cual se construyen cadenas (strings). Ejemplo: Σ = {a, b, c} o Σ = {0, 1}.                                                                          |
| **String**     | Secuencia finita de símbolos tomados de un alfabeto. Ejemplo: si Σ = {a, b}, entonces "aab", "ba", "ε" (cadena vacía) son strings sobre Σ.                                                   |

### 📖 Explicación

Estos son los conceptos fundamentales del procesamiento de lenguaje natural. Un **corpus** es la materia prima (los textos). De un corpus extraemos **tokens** (cada palabra tal como aparece), y el conjunto de tokens únicos forma el **vocabulario** (compuesto de **types**). En la teoría de lenguajes formales, un **alfabeto** es el conjunto base de símbolos y un **string** es cualquier secuencia formada con esos símbolos.

---

## Pregunta 2 — Tokenización y Lematización

**Oración:** _"The cats are chasing the mice in the gardens."_

### 1. Lista de tokens

```
["The", "cats", "are", "chasing", "the", "mice", "in", "the", "gardens", "."]
```

Son **10 tokens** (incluyendo el punto como token separado).

### 2. Lemas de verbos y sustantivos

| Token   | Categoría  | Lema   |
| ------- | ---------- | ------ |
| cats    | Sustantivo | cat    |
| are     | Verbo      | be     |
| chasing | Verbo      | chase  |
| mice    | Sustantivo | mouse  |
| gardens | Sustantivo | garden |

### 3. Caso problemático (edge case) para un tokenizador por espacios

Un tokenizador basado solo en espacios en blanco **fallaría** con contracciones como **`Don't`**. Un tokenizador por espacios lo trataría como un solo token `"Don't"`, cuando en realidad debería separarse en dos tokens: `"Do"` y `"n't"` (o `"Don"` y `"'t"`). Esto es importante porque `"not"` cambia totalmente el significado de la oración y debe tratarse como unidad separada.

Otro ejemplo es **`Week-end`**: un tokenizador por espacios lo trataría como un solo token, cuando podría necesitar separarse en `"Week"` y `"end"`, o mantenerse unido según el contexto.

### 📖 Explicación

La **tokenización** es el proceso de dividir un texto en unidades individuales (tokens). La **lematización** consiste en reducir cada palabra a su forma base o de diccionario (lema): los verbos a infinitivo, los sustantivos a singular, etc. Esto es crucial para que el modelo entienda que "cats" y "cat" se refieren al mismo concepto.

---

## Pregunta 3 — Lenguajes Formales y Strings

**Alfabeto: Σ = {a, b, c}**

### 1. ¿Cuántos strings de longitud 3 se pueden formar?

$$|Σ|^n = 3^3 = 27$$

Se pueden formar **27 strings** de longitud 3.

**Razonamiento:** Para cada posición del string tenemos 3 opciones (a, b o c). Como hay 3 posiciones, multiplicamos: 3 × 3 × 3 = 27.

### 2. Los 10 strings más pequeños en Σ\* (orden lexicográfico)

Los strings en Σ\* se ordenan primero por longitud y luego lexicográficamente:

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

### 📖 Explicación

**Σ\*** (cerradura de Kleene) es el conjunto de **todos los strings posibles** sobre el alfabeto Σ, incluyendo la cadena vacía ε. Se ordenan primero por longitud (más cortos primero) y dentro de la misma longitud, en orden lexicográfico (como en un diccionario). El número total de strings de longitud _n_ siempre es |Σ|^n.

---

## Pregunta 4 — Expresiones Regulares

### 1. Hashtags válidos en un tweet

```python
import re
pattern = r'#[A-Za-z_]\w*'
```

**Explicación:**

- `#` — el símbolo literal de hashtag
- `[A-Za-z_]` — el primer carácter debe ser una letra o guion bajo
- `\w*` — seguido de cero o más caracteres alfanuméricos o guiones bajos (`[A-Za-z0-9_]`)

**Prueba:**

```python
text = "Vean algo de #NLP2024 y #machine_learning hoy"
re.findall(r'#[A-Za-z_]\w*', text)
# Resultado: ['#NLP2024', '#machine_learning']
```

### 2. Fechas en formato DD/MM/YYYY o DD-MM-YYYY

```python
pattern = r'\b\d{2}[/-]\d{2}[/-]\d{4}\b'
```

**Explicación:**

- `\b` — límite de palabra (evita coincidencias parciales)
- `\d{2}` — exactamente 2 dígitos (día)
- `[/-]` — separador: diagonal o guion
- `\d{2}` — exactamente 2 dígitos (mes)
- `[/-]` — separador
- `\d{4}` — exactamente 4 dígitos (año)
- `\b` — límite de palabra

**Prueba:**

```python
text = "Fechas: 18/03/2026 y 25-12-2025"
re.findall(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b', text)
# Resultado: ['18/03/2026', '25-12-2025']
```

### 📖 Explicación

Las **expresiones regulares** son patrones que describen conjuntos de cadenas de texto. Son herramientas fundamentales en NLP para buscar, extraer y validar texto. El módulo `re` de Python permite usar estas expresiones para encontrar patrones en textos.

---

## Pregunta 5 — Modelos de N-gramas

**Corpus:**

- **D1:** "I love NLP."
- **D2:** "I do not love NLP."

### 1. Vocabulario de Bigramas (2-gramas)

Primero tokenizamos (sin puntuación):

- D1: `[I, love, NLP]`
- D2: `[I, do, not, love, NLP]`

Bigramas de D1: `(I, love)`, `(love, NLP)`

Bigramas de D2: `(I, do)`, `(do, not)`, `(not, love)`, `(love, NLP)`

**Vocabulario de bigramas (unión de todos los bigramas únicos):**

| Índice | Bigrama     |
| ------ | ----------- |
| 0      | (I, love)   |
| 1      | (love, NLP) |
| 2      | (I, do)     |
| 3      | (do, not)   |
| 4      | (not, love) |

### 2. Vector de D2 usando el vocabulario de bigramas

Contamos la frecuencia de cada bigrama del vocabulario en D2:

| Bigrama     | ¿Aparece en D2? | Frecuencia |
| ----------- | --------------- | ---------- |
| (I, love)   | No              | 0          |
| (love, NLP) | Sí              | 1          |
| (I, do)     | Sí              | 1          |
| (do, not)   | Sí              | 1          |
| (not, love) | Sí              | 1          |

**Vector de D2:** `[0, 1, 1, 1, 1]`

### 📖 Explicación

Un **n-grama** es una secuencia contigua de _n_ elementos de un texto. Los **bigramas** son pares consecutivos de palabras. Son útiles para capturar contexto local: "not love" tiene un significado muy diferente a "love" solo. El vocabulario de bigramas nos permite representar documentos como vectores numéricos, lo cual es necesario para que los algoritmos de machine learning puedan procesarlos.

---

## Pregunta 6 — TF-IDF

**Datos:**

- Corpus: **1,000 documentos** (N = 1000)
- "transformer" aparece en **10 documentos** (df = 10)
- Documento A: **100 palabras** totales, "transformer" aparece **5 veces**

### 1. Term Frequency (TF)

$$TF(\text{transformer}, A) = 5$$

La frecuencia del término es simplemente el **número de veces** que "transformer" aparece en el documento A = **5**.

### 2. Inverse Document Frequency (IDF)

$$IDF(\text{transformer}) = \log_{10}\left(\frac{N}{df(t)}\right) = \log_{10}\left(\frac{1000}{10}\right) = \log_{10}(100) = 2$$

El IDF es **2**.

### 3. TF-IDF Score

$$TFIDF(\text{transformer}, A) = TF \times IDF = 5 \times 2 = 10$$

El puntaje TF-IDF de "transformer" en el Documento A es **10**.

### 📖 Explicación

**TF-IDF** (Term Frequency – Inverse Document Frequency) es una medida que refleja qué tan **importante** es una palabra para un documento dentro de un corpus:

- **TF** mide qué tan frecuente es la palabra en el documento (más apariciones → más relevante para ese documento).
- **IDF** penaliza las palabras que aparecen en muchos documentos (palabras como "el", "de", "un" tendrán IDF bajo porque aparecen en casi todos los documentos).
- **TF-IDF** combina ambas: una palabra tendrá puntaje alto si aparece mucho en un documento específico pero en pocos documentos del corpus total. Esto indica que es una palabra **distintiva** de ese documento.

---

## Pregunta 7 — TF-IDF en Python (Colab)

```python
# a) Cargar un dataset de texto pequeño
from sklearn.datasets import fetch_20newsgroups

# Cargamos solo 2 categorías para mantenerlo pequeño
categories = ['sci.space', 'rec.sport.baseball']
newsgroups = fetch_20newsgroups(subset='train', categories=categories)
documents = newsgroups.data

print(f"Número de documentos cargados: {len(documents)}")

# b) Preprocesamiento: minúsculas, eliminar puntuación, tokenizar
import re

def preprocess(text):
    text = text.lower()                          # Convertir a minúsculas
    text = re.sub(r'[^\w\s]', '', text)          # Eliminar puntuación
    tokens = text.split()                        # Tokenizar por espacios
    return ' '.join(tokens)                      # Reunir tokens en string

documents_clean = [preprocess(doc) for doc in documents]

print(f"Ejemplo de documento preprocesado:\n{documents_clean[0][:200]}...")

# c) Construir la matriz TF-IDF con scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents_clean)

print(f"\nForma de la matriz TF-IDF: {tfidf_matrix.shape}")

# d) Imprimir resultados
vocabulary = vectorizer.get_feature_names_out()
print(f"\nTamaño del vocabulario: {len(vocabulary)}")

# Vector TF-IDF del primer documento (primeras 10 entradas)
first_doc_vector = tfidf_matrix[0].toarray().flatten()
print(f"\nVector TF-IDF del primer documento (primeras 10 entradas):")
for i in range(10):
    print(f"  {vocabulary[i]}: {first_doc_vector[i]:.4f}")
```

### 📖 Explicación

Este código implementa un pipeline completo de TF-IDF:

1. **Carga de datos:** Usamos `fetch_20newsgroups` de scikit-learn, que contiene posts de grupos de noticias organizados por categoría.
2. **Preprocesamiento:** Convertimos a minúsculas (para que "NLP" y "nlp" se traten igual), eliminamos puntuación y tokenizamos.
3. **TfidfVectorizer:** Esta clase de scikit-learn automatiza el cálculo de TF-IDF. Crea una matriz dispersa donde cada fila es un documento y cada columna es un término del vocabulario.
4. **Resultados:** Mostramos el tamaño del vocabulario y los primeros 10 valores TF-IDF del primer documento.

---

## Pregunta 8 — Word Embeddings con Gensim

```python
# a) Crear word embeddings con Word2Vec
from gensim.models import Word2Vec
from sklearn.datasets import fetch_20newsgroups
import re

# Cargar datos
newsgroups = fetch_20newsgroups(subset='train')
documents = newsgroups.data

# Preprocesar y tokenizar
def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

sentences = [tokenize(doc) for doc in documents]

# Entrenar modelo Word2Vec
# vector_size: dimensión del vector de cada palabra
# window: cuántas palabras de contexto considerar
# min_count: ignorar palabras con frecuencia menor a este valor
# workers: hilos para entrenamiento paralelo
model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=4,
    epochs=10
)

print("Modelo Word2Vec entrenado exitosamente.")
print(f"Vocabulario del modelo: {len(model.wv)} palabras")

# b) 5 palabras más similares a "language"
print("\n5 palabras más similares a 'language':")
similar_words = model.wv.most_similar('language', topn=5)
for word, score in similar_words:
    print(f"  {word}: {score:.4f}")

# c) Similitud coseno entre pares de palabras
print("\nSimilitud coseno:")

sim_cat_dog = model.wv.similarity('cat', 'dog')
print(f"  'cat' y 'dog': {sim_cat_dog:.4f}")

sim_cat_car = model.wv.similarity('cat', 'car')
print(f"  'cat' y 'car': {sim_cat_car:.4f}")

# Interpretación
print(f"\n📊 Interpretación:")
print(f"  'cat'-'dog' ({sim_cat_dog:.4f}) > 'cat'-'car' ({sim_cat_car:.4f})")
print(f"  Esto tiene sentido: 'cat' y 'dog' son animales (contextos similares),")
print(f"  mientras que 'cat' y 'car' son conceptos no relacionados.")
```

### 📖 Explicación

**Word Embeddings** son representaciones vectoriales densas de palabras donde palabras con significados similares tienen vectores cercanos en el espacio:

1. **Word2Vec** aprende estas representaciones analizando los contextos en los que aparecen las palabras. La idea central es: _"Dime con quién andas y te diré quién eres"_ — palabras que aparecen en contextos similares tienen significados similares.

2. **Palabras similares a "language":** El modelo debería devolver palabras relacionadas como "languages", "programming", "english", etc., porque aparecen en contextos similares.

3. **Similitud coseno:** Mide el ángulo entre dos vectores (de -1 a 1):
   - **"cat" y "dog"** deberían tener similitud **alta** porque ambos son animales y aparecen en contextos parecidos ("my cat/dog", "feed the cat/dog").
   - **"cat" y "car"** deberían tener similitud **baja** porque son conceptos de dominios muy diferentes.

> ⚠️ **Nota:** Los resultados exactos dependen del corpus de entrenamiento. Con el corpus de 20 Newsgroups, los resultados pueden variar. Para resultados más robustos, se podrían usar embeddings pre-entrenados como `word2vec-google-news-300`.
