# Word2Vec: Guía Completa del Algoritmo

## 1. Introducción

**Word2Vec** es un algoritmo de aprendizaje profundo desarrollado por Google en 2013 (Tomas Mikolov et al.) que transforma palabras en vectores numéricos densos de baja dimensión. Su objetivo principal es capturar el **significado semántico** de las palabras basándose en su contexto.

La idea fundamental es que **palabras con contextos similares tendrán vectores similares** - esta es la hipótesis distribucional.

---

## 2. Limitaciones de Representaciones Tradicionales

### One-Hot Encoding
- Cada palabra se representa como un vector sparse de dimensión igual al vocabulario
- `gato = [1, 0, 0, 0, 0, ..., 0]`
- **Problemas**:
  - Vector muy grande y sparse
  - No captura relaciones semánticas
  - Todas las palabras son ortogonalmente independientes

### Bag of Words (BoW)
- Cuenta la frecuencia de palabras
- **Problema**: Ignora el orden y el contexto

---

## 3. La Hipótesis Distribucional

> "Deberías conocer una palabra por la compañía que guarda" (J.R. Firth, 1957)

Las palabras que aparecen en contextos similares tienden a tener significados similares. Por ejemplo:
- "perro" y "gato" aparecen cerca de: "mascota", "animal", "casa"
- "rey" y "reina" aparecen cerca de: "corona", "trono", "palacio"

---

## 4. Arquitectura de Word2Vec

Word2Vec ofrece dos arquitecturas principales:

### 4.1 CBOW (Continuous Bag of Words)

```
Contexto: [el, rápido, salta, sobre]
                    ↓
                 CBOW
                    ↓
              "perro" (predicción)
```

**Funcionamiento**:
1. Toma palabras de contexto (ventana)
2. Suma/promedia sus vectores
3. Predice la palabra objetivo

**Formula matemática**:
```
P(w_t | w_{t-1}, w_{t+1}) = softmax(v_{w_t} · \hat{v})
```

**Ventajas**: 
- Mejor con corpus pequeños
- Funciona bien con palabras frecuentes

### 4.2 Skip-gram

```
Palabra objetivo: "perro"
                    ↓
               Skip-gram
                    ↓
Context predicho: [el, rápido, salta, sobre]
```

**Funcionamiento**:
1. Toma una palabra objetivo
2. Para cada palabra de contexto, predice la probabilidad

**Ventajas**:
- Mejor con corpus grandes
- Funciona bien con palabras poco frecuentes

---

## 5. El Modelo Neuronal

### Estructura de una capa

```
Entrada (One-Hot) → Embedding →隐藏层→ Softmax → Salida
   [V×1]            [V×N]       [N×H]    [H×V]
   
V = tamaño del vocabulario
N = dimensión del embedding (típicamente 100-300)
H = tamaño de la capa oculta
```

### Propagación hacia adelante

```
x → W₁ → h → W₂ → y
```

Donde:
- `W₁` es la matriz de embedding de entrada (V×N)
- `W₂` es la matriz de embedding de salida (N×V)
- `h` es la representación oculta

### Pérdida (Softmax)

```
J(θ) = -∑_{c∈Context} log P(w_c | w_t)
```

El softmax normaliza las probabilidades:
```
P(w | context) = exp(v_w · v_context) / ∑_{w'∈V} exp(v_w' · v_context)
```

---

## 6. Técnicas de Optimización

### 6.1 Softmax Jerárquico (Hierarchical Softmax)

**Problema**: Softmax es computacionalmente costoso (O(V))

**Solución**: 
- Crear un árbol de Huffman
- Reducir a O(log V)
- Palabras frecuentes tienen caminos más cortos

### 6.2 Muestreo Negativo (Negative Sampling)

**Idea**: En lugar de predecir todas las palabras del vocabulario, solo predecir algunas muestras negativas.

**Función de pérdida modificada**:
```
J = log σ(v_w · v_c) + ∑_{i=1}^{k} E_{w_i~P_n}[log σ(-v_{w_i} · v_c)]
```

**Estrategia de muestreo**: 
- Distribución unigram elevada a 3/4
- `P(w) = count(w)^{3/4} / ∑ count^{3/4}`

### 6.3 Subsampling de Palabras Frecuentes

**Problema**: Palabras como "el", "de", "la" aparecen mucho y aportan poco.

**Solución**: Eliminar palabras con probabilidad:
```
P(w) = 1 - √(t / f(w))
```
Donde `t` es un threshold (típicamente 10^-5)

---

## 7. Propiedades del Embedding

### 7.1 Operaciones Vectoriales

El magia de Word2Vec: **aritmética de palabras**

```
vec("rey") - vec("hombre") + vec("mujer") ≈ vec("reina")
vec("París") - vec("Francia") + vec("Alemania") ≈ vec("Berlín")
```

### 7.2 Analogías

| Analogía | Operación |
|----------|-----------|
| hombre→rey | mujer→? |
| París→Francia | Berlín→? |
| comer→comió | caminar→? |

### 7.3 Similitud Coseno

```
sim(A, B) = (A · B) / (||A|| · ||B||)
```

Valores cercanos a 1 = muy similares

---

## 8. Hiperparámetros Importantes

| Parámetro | Valor Típico | Efecto |
|-----------|--------------|--------|
| Dimensión (size) | 100-300 | Más dimensiones = más relaciones |
| Ventana (window) | 5-10 | Mayor = más contexto |
| Learning rate | 0.01-0.025 | Afecta convergencia |
| Epochs | 5-20 | Más epochs = mejor training |
| Min count | 5-10 | Filtra palabras raras |
| Negative samples | 5-20 | Trade-off calidad/velocidad |

---

## 9. Implementación en Python ( Gensim )

```python
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

# Corpus tokenizado
sentences = [
    ['el', 'perro', 'ladra'],
    ['el', 'gato', 'maulla'],
    ['los', 'perros', 'ladran'],
    # ... más oraciones
]

# Entrenamiento CBOW
model = Word2Vec(
    sentences=sentences,
    vector_size=100,      # dimensión del embedding
    window=5,             # tamaño de ventana
    min_count=1,          # frecuencia mínima
    workers=4,            # hilos paralelo
    sg=0,                 # 0=CBOW, 1=Skip-gram
    epochs=100            # épocas de entrenamiento
)

# Obtener vector de una palabra
vector = model.wv['perro']

# Encontrar palabras similares
similares = model.wv.most_similar('perro', topn=5)

# Analogías
analogia = model.wv.most_similar(positive=['rey', 'mujer'], negative=['hombre'])
```

---

## 10. Ventajas y Desventajas

### Ventajas
- ✅ Captura relaciones semánticas y sintácticas
- ✅ Vectores densos y de baja dimensión
- ✅ Pre-entrenamiento eficiente
- ✅ Operaciones vectoriales con significado

### Desventajas
- ❌ No tiene en cuenta el orden de las palabras (como RNN/LSTM)
- ❌ No puede manejar palabras fuera del vocabulario (OOV)
- ❌ Asigna un solo vector por palabra (polisemia)
- ❌ No considera estructura morfológica

---

## 11. Extensiones y Modelos Relacionados

### glove (Global Vectors)
- Combina estadísticas de co-ocurrencia global con Word2Vec
- Stanford NLP

### FastText
- Considera sub-palabras (n-gramas)
- Maneja mejor palabras OOV
- Facebook AI

### BERT / ELMo
- Contextuales (cada palabra tiene vector diferente según contexto)
- Basados en Transformers
- Estado del arte actual

---

## 12. Métricas de Evaluación

### Intrínsecas
- **Analogías**: `vec(a) - vec(b) + vec(c) ≈ vec(d)`
- **Similitud**: Correlación con juicios humanos

### Extrínsecas
- Clasificación de texto
- Named Entity Recognition (NER)
- Análisis de sentimiento

---

## 13. Ejemplo Visual Completo

```
Corpus: "El perro juega con la pelota. El gato duerme en el sofá."

Paso 1: Tokenización
[[el, perro, juega, con, la, pelota], [el, gato, duerme, en, el, sofa]]

Paso 2: Crear pares (ventana=2, skip-gram)
(perro, el), (perro, juega), (perro, con), (perro, la)
(gato, el), (gato, duerme), (gato, en), (gato, el)
...

Paso 3: Entrenar red neuronal
Entrada: [0,1,0,0,...] (perro)
Salida: distribución de probabilidad sobre vocabulario

Paso 4: Extraer embeddings
Matriz de embedding: cada fila = vector de una palabra

Paso 5: Usar embeddings
similitud("perro", "gato") → 0.85
similitud("perro", "pelota") → 0.42
```

---

## 14. Conclusión

Word2Vec revolucionó el NLP al demostrar que **es posible aprender representaciones densas de palabras** que capturan relaciones semánticas. Su impacto sentó las bases para modelos más avanzados como BERT y GPT.

La clave de su éxito: **aprovechar la hipótesis distribucional** - palabras en contextos similares tienen significados similares.

---

## Referencias

- Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
- Mikolov, T., et al. (2013). "Distributed Representations of Words and Phrases and their Compositionality"
- Rong, X. (2014). "word2vec Parameter Learning Explained"
