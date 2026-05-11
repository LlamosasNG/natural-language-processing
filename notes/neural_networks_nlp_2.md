# Guía de Estudio: Redes Neuronales para Procesamiento de Lenguaje Natural (NLP)

Este documento presenta una guía de estudio integral sobre redes neuronales aplicadas al Procesamiento de Lenguaje Natural, basada en el documento original "Neural networks for NLP".

---

## 1. Fundamentos: Funciones de Activación

Las funciones de activación son componentes esenciales que introducen no-linealidad en las redes neuronales, permitiendo que aprendan patrones complejos.

### 1.1 ReLU (Rectified Linear Unit)

**Definición:**
```
ReLU(x) = max(0, x)
```

**Derivada:**
```
dReLU(x)/dx = 0 si x < 0
              1 si x > 0
```

> **Nota técnica:** En x = 0, la derivada no está definida, pero en la práctica se establece en 0 o 1 (la mayoría de las bibliotecas usan 0).

**Intuición:**
- Si la entrada es negativa, la salida es 0
- Si la entrada es positiva, la salida es la entrada misma

**Ventajas:**
- Introduce no-linealidad, permitiendo que las redes aprendan patrones complejos
- Computacionalmente económica (solo comparaciones)
- Ayuda a reducir el problema del gradiente desvaneciente comparado con sigmoid o tanh

**Desventaja común:**
- **"ReLUs Muertas"**: Las neuronas pueden quedar/atascadas emitiendo 0 si los pesos empujan las entradas negativas todo el tiempo

### 1.2 Softmax

**Definición:**
Para un vector de puntuaciones z₁, ..., z_K:
```
softmax(zᵢ) = eᶻⁱ / Σₖ₌₁ eᶻᵏ
```

**Intuición:**
- Convierte puntuaciones brutas ("logits") en probabilidades
- Todos los valores están entre 0 y 1
- Los valores suman 1

**Ejemplo práctico:**
Si el modelo produce: `[2.0, 1.0, 0.1]`
Softmax lo transforma en: `[0.66, 0.24, 0.10]`

**Importancia:** Softmax casi siempre se combina con la función de pérdida de entropía cruzada. En este caso, el gradiente se simplifica dramáticamente:
```
∂L/∂zᵢ = ŷᵢ - yᵢ
```
Donde:
- ŷᵢ = salida de softmax
- yᵢ = etiqueta verdadera (one-hot)

### 1.3 Resumen de Funciones de Activación

| Función | Uso Principal |
|---------|---------------|
| **Sigmoid** | Salida tipo probabilidad |
| **ReLU** | Capas ocultas (simplicidad, evita gradientes desvanecientes) |
| **Softmax** | Clasificación multiclase (capa de salida) |

---

## 2. Entrenamiento de Redes Neuronales

### 2.1 Propagación Hacia Adelante (Forward Propagation)

La propagación hacia adelante es el proceso mediante el cual la información fluye desde la entrada hasta la salida a través de las capas de la red:

```
Z[1] = W[1]x + b[1]
a[1] = ReLU(Z[1])
Z[2] = W[2]a[1] + b[2]
ŷ = a[2] = g(Z[2])
```

Donde:
- **x**: vector de entrada
- **W**: matriz de pesos
- **b**: vector de sesgo (bias)
- **g**: función de activación
- **ŷ**: predicción de la red

### 2.2 Funciones de Pérdida

Para clasificación, se utiliza la **pérdida de entropía cruzada**:
```
L = -Σ y log(ŷ)
```

El objetivo es minimizar la pérdida sobre los datos de entrenamiento.

### 2.3 Descenso de Gradiente

El objetivo es cambiar los pesos para mejorar las predicciones. Para esto, se calcula cuánto contribuyó cada peso al error.

### 2.4 Retropropagación (Backpropagation)

**Definición:** Aplicación de la regla de la cadena del cálculo infinitesimal. El error fluye hacia atrás, capa por capa.

**Concepto clave:** La retropropagación es una contabilidad eficiente para derivadas parciales.

**Fórmula general:**
```
∂L/∂W[2] = ∂L/∂ŷ × ∂ŷ/∂W[2]
∂L/∂W[1] = ∂L/∂z[1] × ∂z[1]/∂W[1]
```

---

## 3. ¿Por qué las Redes Neuronales Funcionan Bien en NLP?

### 3.1 Ventajas Clave

| Característica | Descripción |
|----------------|-------------|
| **Entradas sparse de alta dimensión** | Palabras, n-gramas |
| **Representaciones continuas** | Embeddings densos |
| **Aprendizaje de características** | Reemplaza la ingeniería manual de características |

### 3.2 Casos de Uso en NLP

- **Análisis de sentimiento**: Clasificación de opiniones positivas/negativas
- **Etiquetado de partes del discurso (POS tagging)**: Categorización gramatical
- **Reconocimiento de entidades nombradas (NER)**: Identificación de nombres propios, organizaciones, etc.
- **Embeddings de palabras**: Representaciones vectoriales densas

### 3.3 Limitaciones Importantes

- **Hambrientos de datos**: Requieren grandes cantidades de información
- **Difíciles de interpretar**: "Cajas negras"
- **Propensos al sobreajuste**: Necesitan regularización

---

## 4. Arquitecturas de Redes Neuronales

### 4.1 Redes Neuronales Feed-Forward (FFNN)

Son el modelo fundacional básico. La información fluye en una dirección: de la entrada hacia la salida, sin ciclos ni retroalimentación.

### 4.2 Redes Neuronales Convolucionales (CNNs)

**Idea Central:**
Las CNNs están diseñadas para procesar datos estructurados en cuadrícula (grid-structured) aprendizaje de patrones locales a través de filtros convolucionales.

**Tipos de Datos Comunes:**
- Imágenes
- Video
- Texto representado como embeddings (convoluciones 1D)

**¿Por qué funcionan en NLP?**
Tratan el texto como secuencias estructuradas, capturando patrones de n-gramas locales.

**Ejemplo NLP - Clasificación de oraciones:**
- **Tarea:** Detección de spam o clasificación de temas
- **Enfoque:** Aplicar convoluciones sobre embeddings de palabras
- **Framework:** Keras Conv1D

**Cuándo usarlas:** Cuando el contexto local es más importante que las dependencias de largo alcance.

---

### 4.3 Redes Neuronales Recurrentes (RNNs)

**Idea Central:**
Las RNNs procesan secuencias manteniendo un estado oculto que evoluciona con el tiempo, permitiendo que el modelo capture información depende del orden.

**Tipos de Datos Comunes:**
- Series temporales
- Texto
- Señales de voz

**Ejemplo NLP - Modelado de lenguaje a nivel de carácter:**
- **Tarea:** Predecir el siguiente carácter en una secuencia
- **Resultado:** Aprenden patrones de ortografía y sintaxis
- **Limitación:** Dificultad con contexto a largo plazo

**Nota histórica:** Las RNNs son históricamente importantes en NLP pero han sido reemplazadas en gran medida por LSTM, GRU y Transformers.

---

### 4.4 Redes de Memoria Corto-Plazo Largo (LSTM)

**Idea Central:**
Las LSTM introducen mecanismos de puertas (gates) que regulan cómo se almacena, olvida y expone la información, permitiendo el aprendizaje sobre secuencias largas.

**Componentes de una LSTM:**
1. **Puerta de entrada (Input Gate):** Controla qué información nueva se almacena
2. **Puerta de olvido (Forget Gate):** Decide qué información descartar
3. **Puerta de salida (Output Gate):** Determina qué información se emite

**Ejemplo NLP - Análisis de sentimiento de documentos largos:**
- **Tarea:** Clasificar sentimiento en reseñas de varios párrafos
- **Ventaja:** Mantiene contexto a través de muchas oraciones
- **Framework:** Keras Embedding + LSTM

**Contexto histórico:** Antes de los Transformers, las LSTM eran la arquitectura dominante en NLP para modelado de secuencias.

---

### 4.5 Unidades Recurrentes Puertizadas (GRU)

**Idea Central:**
Las GRUs simplifican las LSTM combinando puertas, resultando en menos parámetros y entrenamiento más rápido, manteniendo una fuerte capacidad de modelado de secuencias.

**Diferencia con LSTM:**
- No tienen puerta de salida
- Combinan la puerta de entrada y la puerta de olvido en una sola puerta de actualización

**Ejemplo NLP - Traducción automática neuronal (modelos tempranos):**
- **Tarea:** Traducir oraciones entre idiomas
- **Ventaja:** Alto rendimiento con menos recursos que LSTM
- **Usado en:** Sistemas encoder-decoder seq2seq tempranos

**Cuándo usarlas:** Cuando la eficiencia del modelo es crítica.

---

### 4.6 Autoencoders

**Idea Central:**
Los autoencoders aprenden representaciones compactas y significativas de los datos reconstructyendo las entradas a través de una capa cuello de botella (bottleneck).

**Usos Típicos:**
- Compresión
- Reducción de ruido
- Detección de anomalías

**Ejemplo NLP - Embeddings de oraciones y reducción de dimensionalidad:**
- **Tarea:** Comprimir vectores TF-IDF o embeddings
- **Caso de uso:** Clustering o similitud semántica
- **Variante:** Autoencoders seq2seq usando LSTM

**Ventaja en NLP:** Útiles cuando las etiquetas son escasas o las representaciones necesitan aprenderse sin supervisión.

---

### 4.7 Redes Neuronales de Grafos (GNNs)

**Idea Central:**
Las GNNs operan sobre grafos, aprendiendo representaciones que combinan características de nodos con estructura relacional.

**Tipos de Datos Comunes:**
- Redes sociales
- Moléculas
- Grafos de conocimiento

**Ejemplo NLP - Grafos semánticos y NLP basado en conocimiento:**
- **Tarea:** Extracción de relaciones o问答 (question answering)
- **Enfoque:** Aplicar GNNs a árboles de dependencia o grafos de conocimiento
- **Ejemplo:** Entity linking usando propagación en grafos

**Ventaja:** Conecta estructura simbólica y aprendizaje neuronal en NLP.

---

### 4.8 Redes Generativas Adversariales (GANs)

**Idea Central:**
Las GANs consisten en dos redes en competencia:
- **Generador:** Produce datos sintéticos
- **Discriminador:** Distingue lo real de lo falso

El entrenamiento es un juego minimax entre los dos modelos.

**Ejemplo NLP - Aumento de datos textuales y entrenamiento adversarial:**
- **Tarea:** Generar paráfrasis u oraciones sintéticas
- **Desafío:** Los tokens de texto discretos hacen difícil el entrenamiento de GANs
- **Soluciones:** SeqGAN, GANs basados en Aprendizaje por Refuerzo

**Nota:** Las GANs son menos comunes en NLP que en visión, pero son valiosas para aumento de datos y pruebas de robustez.

---

## 5. Resumen Comparativo de Arquitecturas

| Arquitectura | Fortaleza en NLP |
|---------------|------------------|
| **CNN** | Características locales de n-gramas |
| **RNN** | Orden secuencial |
| **LSTM/GRU** | Contexto de largo alcance |
| **Autoencoder** | Representaciones no supervisadas |
| **GNN** | Conocimiento lingüístico estructurado |
| **GAN** | Generación y aumento de datos |

---

## 6. Conclusión Final

**Diferentes arquitecturas neurales existen porque diferentes estructuras de datos demandan diferentes sesgos inductivos:**

- Las arquitecturas modernas de NLP a menudo combinan arquitecturas
- **El Transformer se ha convertido en el backbone central** para la mayoría de los modelos de estado del arte

Esta guía proporciona una base sólida para comprender las arquitecturas fundamentales que sustentan el Procesamiento de Lenguaje Natural moderno.

---

## 7. Glosario de Términos Técnicos

| Término | Definición |
|---------|------------|
| **Backpropagation** | Algoritmo de aprendizaje que calcula gradientes propagando el error hacia atrás |
| **Bidirectional LSTM** | LSTM que procesa la secuencia en ambas direcciones |
| **Bottleneck layer** | Capa intermedia con menos neuronas que fuerza la compresión |
| **Cross-entropy loss** | Función de pérdida común para clasificación |
| **Embedding** | Representación vectorial densa de palabras u otros tokens |
| **Encoder-decoder** | Arquitectura para transformar secuencias de entrada en secuencias de salida |
| **Feed-forward** | Arquitectura donde la información fluye en una dirección |
| **Gates** | Mecanismos que controlan el flujo de información en LSTM/GRU |
| **Gradient descent** | Algoritmo de optimización para minimizar la función de pérdida |
| **Hidden state** | Representación interna de la red que captura información de la secuencia |
| **Inductive bias** | Supuestos previos que guían el aprendizaje |
| **Logits** | Puntuaciones brutas antes de aplicar softmax |
| **Loss function** | Función que mide el error de la predicción |
| **One-hot encoding** | Representación donde solo un valor es 1 y los demás son 0 |
| **Overfitting** | Cuando el modelo aprende ruido en lugar de patrones generales |
| **Regularization** | Técnicas para prevenir el sobreajuste |
| **ReLU** | Función de activación que devuelve max(0, x) |
| **Softmax** | Función que convierte logits en probabilidades |
| **Transformer** | Arquitectura basada en atención, dominante en NLP moderno |
| **Vanishing gradient** | Problema donde los gradientes se hacen muy pequeños para aprender |

---

*Guía de estudio generada a partir del documento "Neural networks for NLP"*