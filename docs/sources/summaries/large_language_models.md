# Modelos de Lenguaje de Gran Escala (LLMs)

## Origen del Término "Modelo de Lenguaje"

El término surgió en el reconocimiento del habla. La figura clave fue **Fred Jelinek** del grupo de investigación de IBM en Yorktown Heights, Nueva York, durante finales de los años 1970 y 1980.

**Problema original**: Dada una entrada acústica ambigua, elegir la secuencia de palabras más probable.

**Formalización**:
$$\hat{W} = \arg\max_W P(A|W) \cdot P(W)$$

Donde $P(A|W)$ es el modelo acústico.

## Definición Moderna

Un modelo de lenguaje asigna una probabilidad a una secuencia de tokens:
$$P(w_1, w_2, ..., w_n)$$

Usando la regla de la cadena:
$$P(w_1^n) = \prod_{t=1}^{n} P(w_t | w_1^{t-1})$$

**Interpretación**: "¿Qué tan sorprendidos deberíamos estar por el siguiente token, dado los anteriores?"

## Estimación de Máxima Verosimilitud (MLE)

**Idea central**: Elegir los parámetros del modelo que hagan que los datos de entrenamiento observados sean lo más probables posible.

$$\hat{\theta} = \arg\max_\theta P(D|\theta)$$

Para modelado de lenguaje:
$$\hat{\theta} = \arg\max_\theta \prod_{t=1}^{n} P_\theta(w_t | w_1^{t-1})$$

Usando logaritmos (más conveniente):
$$\hat{\theta} = \arg\max_\theta \sum_{t=1}^{n} \log P_\theta(w_t | w_1^{t-1})$$

## Modelos N-gram

**Aproximación de Markov**: El condicionamiento exacto es imposible → se aproxima:
$$P(w_t | w_1^{t-1}) \approx P(w_t | w_{t-n+1}^{t-1})$$

- **Bigram**: $P(w_t | w_{t-1})$
- **Trigram**: $P(w_t | w_{t-2}, w_{t-1})$

**Estimación MLE**:
$$P(w_t | w_{t-1}) = \frac{count(w_{t-1}, w_t)}{count(w_{t-1})}$$

### Problema de Esparacidad

- Muchos n-grams nunca aparecen
- Llevan a probabilidades cero
- Una oración con bigram no visto ⇒ probabilidad total = 0

### Suavizado (Smoothing)

**Ejemplo: Add-one (Laplace)**

$$P(w|like) = \frac{count(like \cdot w) + 1}{count(like) + V}$$

**Idea clave**: Tomar un poco de masa de probabilidad de eventos frecuentes para permitir los no vistos.

## Evaluación: Perplexidad

$$PP = P(w_1^n)^{-1/n}$$

**Interpretación**:
- PP = 10: el modelo elige entre 10 opciones igualmente probables en cada paso
- PP = 100: modelo mucho peor
- Menor perplexity = menos sorpresa
- "Factor promedio de ramificación"

## Limitaciones de los N-grams

- Contexto fijo
- Esparacidad
- No hay similitud semántica
- Explosión de vocabulario
- "Dogs" y "cats" no están relacionados a menos que se vean juntos explícitamente

## Modelos de Lenguaje Neurales

**Bengio et al. (2003)**: Primer modelo neural de lenguaje.
- Palabras de contexto → Embeddings → Capa oculta → Softmax → $P(w_t | contexto)$

## ELMo (2018)

**Autores**: Peters et al., Allen Institute for AI (AI2)

**Ideas clave**:
- Modelo de lenguaje bidireccional profundo
- Representaciones contextuales de palabras
- Diferentes vectores para la misma palabra en diferentes contextos

**Ejemplo**: "river bank" vs. "investment bank"

## Cómo Entrenar un LLM

### Paso 1: Datos
- Corpus de texto grandes: libros, Wikipedia, noticias, texto web
- Escala: miles de millones a billones de tokens

### Paso 2: Tokenización
- Tokenización subword (ej. BPE, WordPiece)
- Maneja palabras raras y no vistas eficientemente

### Paso 3: Objetivo de Entrenamiento (MLE)
Objetivo autorregresivo: predecir cada token a partir de los tokens anteriores.

**Ejemplo**:
| Input | Target |
|-------|--------|
| \<s\> | The |
| \<s\> The | cat |
| \<s\> The cat | sat |
| ... | ... |
| \<s\> The cat sat on the mat | \</s\> |

### Paso 4: Optimización
- Descenso de gradiente estocástico
- Mini-batches
- Muchas épocas (pasadas sobre los datos)
- No se necesitan datos etiquetados más allá del texto sin procesar

### Paso 5: Evaluación
- Conjunto de prueba (held-out)
- Métrica: **Perplexidad**
- Menor perplexity = mejor generalización

### Usos de ejemplo

- **Análisis de sentimiento**: $P(Positive|s) >>> P(Negative|s)$
- **Trivia**: $P(Cervantes|s) >>> P(Dickens|s)$
- **Respuesta a preguntas**: $P(B|s) >>> P(Error|s)$