# 📝 Examen de Práctica — Fundamentos de PLN

**Fecha:** miércoles, 18 de marzo de 2026

---

## Pregunta 1 — Definiciones Fundamentales

Define los siguientes términos:

| Término         | Definición |
| --------------- | ---------- |
| **Corpus**      |            |
| **Vocabulario** |            |
| **Token**       |            |
| **Tipo**        |            |
| **Alfabeto**    |            |
| **Cadena**      |            |

---

## Pregunta 2 — Tokenización y Lematización

Dada la siguiente oración:

> _"The cats are chasing the mice in the gardens."_

1. Lista los **tokens**.
2. Lista los **lemas** de todos los verbos y sustantivos de la oración.
3. Explica un **caso borde** donde un tokenizador simple por espacios en blanco fallaría.

> 💡 **Pista — ejemplos de casos borde:** `Don't`, `Week-end`

---

## Pregunta 3 — Lenguajes Formales y Cadenas

Sea el alfabeto **Σ = {a, b, c}**.

1. ¿Cuántas cadenas de **longitud 3** se pueden formar sobre Σ?
2. Lista las **10 cadenas más pequeñas** en Σ\* (orden lexicográfico).

---

## Pregunta 4 — Expresiones Regulares

Escribe **expresiones regulares compatibles con Python** (módulo `re`) para encontrar:

1. Todos los **hashtags válidos** en un tweet
   - Ejemplos: `#NLP2024`, `#machine_learning`

2. **Fechas** en formato `DD/MM/YYYY` o `DD-MM-YYYY`

---

## Pregunta 5 — Modelos de Lenguaje N-gramas

Considera el siguiente corpus de juguete:

- **D1:** "I love NLP."
- **D2:** "I do not love NLP."

1. Construye el **vocabulario de Bigramas (2-gramas)** para este corpus.
2. Representa **D2** como un vector usando el vocabulario de bigramas.

---

## Pregunta 6 — TF-IDF

Tienes un corpus de **1,000 documentos**.

- La palabra **"transformer"** aparece en **10 documentos**.
- El **Documento A** tiene **100 palabras en total**, y "transformer" aparece **5 veces**.

### Tareas

1. Calcula la **Frecuencia de Término (TF)** de "transformer" en el Documento A.
2. Calcula la **Frecuencia Inversa de Documento (IDF)** usando el logaritmo en base 10.
3. Calcula el **puntaje TF-IDF final** de "transformer" en el Documento A.

### Fórmulas

**Frecuencia de Término:**

$$TF(t, d) = \text{número de ocurrencias del término } t \text{ en el documento } d$$

$$TF(\text{transformer} $$

**Frecuencia Inversa de Documento:**

$$IDF(t) = \log_{10}\left(\frac{N}{df(t)}\right)$$

- $N$ = Número total de documentos
- $df(t)$ = Número de documentos que contienen el término

**TF-IDF:**

$$TFIDF(t, d) = TF(t, d) \times IDF(t)$$

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

---

## Pregunta 8 — Embeddings de Palabras con Gensim

Usando `gensim`, realiza lo siguiente:

**a)** Carga o crea **embeddings de palabras**

- Usa Word2Vec de `gensim` entrenado sobre un texto pequeño.

**b)** Muestra las **5 palabras más similares** a la palabra `"language"`.

**c)** Calcula la **similitud coseno** entre:

- `"cat"` y `"dog"`
- `"cat"` y `"car"`
