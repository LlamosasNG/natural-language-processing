# 📘 Guía de Estudio: Redes Neuronales para PLN

**Documento original:** Neural networks for NLP.pdf  
**Fecha de elaboración:** 1 de mayo de 2026  
**Nivel:** Intermedio-Avanzado

---

## 1. Introducción y Conceptos Fundamentales

### ¿Qué es una Red Neuronal?

Una **red neuronal** es un modelo computacional inspirado en el funcionamiento del cerebro biológico, compuesto por capas de nodos interconectados que procesan información mediante propagación de señales.

**Referencia:** https://mriquestions.com/what-is-a-neural-network.html

---

## 2. Funciones de Activación

Las **funciones de activación** introducen no-linealidad en las redes neuronales, permitiendo aprender patrones complejos.

### 2.1 ReLU (Rectified Linear Unit)

**Definición:**
```
ReLU(x) = max(0, x)
```

**Derivada:**
- 0 si x < 0
- 1 si x > 0

**En x = 0:** La derivada es indefinida, pero en práctica se establece como 0 o 1 (la mayoría de librerías usan 0).

| Característica | Descripción |
|----------------|-------------|
| **Intuición** | Si el input es negativo → output = 0; Si es positivo → output = el input mismo |
| **Ventajas** | Computacionalmente barata, evita el problema de *vanishing gradients* |
| **Desventaja** | *Dead ReLUs*: neuronas pueden quedar "atascadas" en 0 si los pesos empujan los inputs a negativos constantemente |
| **Uso típico** | Capas ocultas |

**Analogía de Ingeniería:** Piensa en ReLU como un **diodo en electrónica**: solo permite el paso de corriente (información) en una dirección. Si el voltaje es negativo, el circuito se "apaga" (output = 0). Si es positivo, la señal pasa directamente.

---

### 2.2 Sigmoid

| Característica | Descripción |
|----------------|-------------|
| **Output** | Probabilidad (entre 0 y 1) |
| **Uso histórico** | Popular en redes tempranas |
| **Problema** | Sufre de *vanishing gradients* |

---

### 2.3 Softmax

**Definición:** Para un vector de puntuaciones z₁, ..., zₖ:

```
softmax(zᵢ) = e^zᵢ / Σⱼ₌₁ᴷ e^zⱼ
```

| Característica | Descripción |
|----------------|-------------|
| **Propósito** | Convertir *logits* (puntuaciones brutas) en probabilidades |
| **Propiedad** | Todos los valores están entre 0 y 1, y suman 1 |
| **Uso típico** | Capa de salida en clasificación multiclase |
| **Se combina con** | *Cross-entropy loss* |

**Ejemplo Práctico:**

```
Salida del modelo: [2.0, 1.0, 0.1]
Después de Softmax: [0.66, 0.24, 0.10]
```

**Analogía de Ingeniería:** Softmax es como un **sistema de distribución de presupuesto**: tienes $100 (probabilidad total = 1) y debes asignarlos entre diferentes departamentos (clases). Las puntuaciones brutas determinan qué tanto recibe cada uno, pero el total siempre debe ser $100.

---

### Resumen: ReLU vs Softmax

| Función | Propósito |
|---------|-----------|
| **ReLU** | Decide si una neurona está activa |
| **Softmax** | Decide qué clase es más probable |

---

### ¿Qué pasa si removemos la función de activación?

Sin funciones de activación, la red neuronal se reduce a una **regresión lineal**, sin importar cuántas capas tenga. Esto elimina la capacidad de aprender patrones no lineales complejos.

---

## 3. Entrenamiento de Redes Neuronales

### 3.1 Forward Propagation (Propagación Hacia Adelante)

```
Z[1] = W[1]x + b[1]     # Combinación lineal capa 1
a[1] = ReLU(Z[1])       # Activación capa 1
z[2] = W[2]a[1] + b[2]  # Combinación lineal capa 2
ŷ = a[2] = g(z[2])      # Predicción final
```

---

### 3.2 Funciones de Pérdida (*Loss Functions*)

**Para clasificación → Cross-Entropy Loss:**

```
L = -Σ y log(ŷ)
```

**Objetivo:** Minimizar la pérdida sobre los datos de entrenamiento.

---

### 3.3 Gradient Descent

| Paso | Descripción |
|------|-------------|
| 1 | Queremos cambiar los pesos para mejorar la predicción |
| 2 | Computamos cuánto contribuyó cada peso al error |
| 3 | Ajustamos pesos en dirección opuesta al gradiente |

---

### 3.4 Backpropagation (Retropropagación)

| Concepto | Explicación |
|----------|-------------|
| **Base matemática** | Regla de la cadena (*chain rule*) del cálculo |
| **Mecanismo** | El error fluye hacia atrás, capa por capa |
| **Propósito** | Calcular cuánto contribuyó cada peso al error |
| **Esencia** | *Bookkeeping* eficiente de derivadas parciales |

**Fórmulas clave:**
```
∂L/∂W[2] = (∂L/∂ŷ) × (∂ŷ/∂W[2])
∂L/∂W[1] = (∂L/∂z[1]) × (∂z[1]/∂W[1])
```

**Con Softmax + Cross-Entropy, el gradiente se simplifica:**
```
∂L/∂zᵢ = ŷᵢ - yᵢ
```

Donde:
- ŷᵢ = salida de softmax
- yᵢ = etiqueta verdadera (*one-hot*)

---

## 4. ¿Por Qué las Redes Neuronales Funcionan Bien para PLN?

| Ventaja | Explicación |
|---------|-------------|
| **Inputs dispersos de alta dimensión** | Palabras y n-grams se representan eficientemente |
| **Representaciones continuas** | Los *embeddings* capturan similitud semántica |
| **Feature learning automático** | Reemplaza la ingeniería manual de características |

---

### Casos de Uso en NLP

- ✅ **Análisis de sentimiento**
- ✅ **POS Tagging** (etiquetado gramatical - extendido luego a RNNs)
- ✅ **Reconocimiento de entidades nombradas** (NER)
- ✅ **Word Embeddings** (tema de próximos capítulos)

---

### Limitaciones Importantes

| Limitación | Implicación |
|------------|-------------|
| **Data-hungry** | Requieren grandes volúmenes de datos |
| **Difícil interpretación** | "Caja negra" - difícil de debuggear |
| **Propenso a overfitting** | Necesita regularización adecuada |

---

## 5. Arquitecturas de Redes Neuronales

> **Nota del documento:** Varias arquitecturas neuronales han sido propuestas. Ya cubrimos el modelo fundacional, actualmente conocido como **Feed-Forward Neural Networks (FFNNs)**. Existen arquitecturas especializadas diseñadas para abordar desafíos específicos como estructura espacial, datos secuenciales, aprendizaje de representaciones y generación.

---

### 📊 Tabla Comparativa Resumen

| Arquitectura | Fortaleza en NLP | Mejor Para |
|--------------|------------------|------------|
| **CNN** | Features locales de n-grams | Clasificación de textos cortos |
| **RNN** | Orden secuencial | Secuencias con dependencia temporal |
| **LSTM / GRU** | Contexto de largo alcance | Documentos extensos, traducción |
| **Autoencoder** | Representaciones no supervisadas | Compresión, clustering semántico |
| **GNN** | Conocimiento lingüístico estructurado | Grafos de dependencia, QA |
| **GAN** | Generación y augmentación de datos | Paráfrasis, datos sintéticos |

> **Takeaway final:** Los sistemas modernos de NLP a menudo combinan arquitecturas, pero el **Transformer** se ha convertido en la columna vertebral central para la mayoría de modelos state-of-the-art.

---

### 5.1 CNN (Convolutional Neural Networks)

#### Idea Central

Las CNNs están diseñadas para procesar datos estructurados en grilla aprendiendo **patrones locales** mediante filtros convolucionales. Aunque originalmente fueron desarrolladas para visión, las CNNs han demostrado utilidad en NLP tratando el texto como secuencias estructuradas.

**Referencias:**
- https://medium.com/@bdhuma/6-basic-things-to-know-about-convolution-daef5e1bc411
- https://developersbreach.com/convolution-neural-network-deep-learning/

| Aspecto | Detalle |
|---------|---------|
| **Datos comunes** | Imágenes, video, texto representado como embeddings (convoluciones 1D) |
| **Implementación popular** | ResNet usando PyTorch / TensorFlow |
| **Fortaleza** | Contexto local > dependencias de largo alcance |

---

#### Ejemplo No-NLP: Clasificación de Imágenes Médicas

| Campo | Detalle |
|-------|---------|
| **Modelo** | ResNet-50 |
| **Tarea** | Detectar neumonía en rayos X de tórax |
| **Por qué CNNs** | Aprenden características visuales jerárquicas automáticamente |

---

#### Ejemplo NLP: Clasificación de Oraciones

| Campo | Detalle |
|-------|---------|
| **Tarea** | Detección de spam o clasificación de temas |
| **Enfoque** | Aplicar convoluciones sobre word embeddings para capturar patrones locales de n-grams |
| **Framework** | Keras Conv1D |

---

**Analogía de Ingeniería:** Una CNN es como un **scanner de código de barras** que se desliza sobre el texto. En cada posición, lee una ventana pequeña (n-gram) y extrae características. No necesita ver todo el documento a la vez, sino patrones locales que son indicativos de la clase.

---

### 5.2 RNN (Recurrent Neural Networks)

#### Idea Central

Las RNNs procesan secuencias manteniendo un **estado oculto** que evoluciona en el tiempo, permitiendo al modelo capturar información dependiente del orden.

**Referencia:** https://www.geeksforgeeks.org/data-analysis/difference-between-feed-forward-neural-networks-and-recurrent-neuralnetworks/

| Aspecto | Detalle |
|---------|---------|
| **Datos comunes** | Series temporales, texto, señales de habla |
| **Implementación popular** | SimpleRNN en TensorFlow/Keras |
| **Limitación** | Dificultad con contexto de largo plazo |

---

#### Ejemplo No-NLP: Predicción de Series Temporales

| Campo | Detalle |
|-------|---------|
| **Tarea** | Predecir carga eléctrica o temperatura |
| **Por qué RNNs** | Las dependencias temporales importan |

---

#### Ejemplo NLP: Modelado de Lenguaje a Nivel de Carácter

| Campo | Detalle |
|-------|---------|
| **Tarea** | Predecir el siguiente carácter en una secuencia |
| **Resultado** | Aprende patrones de ortografía y sintaxis |
| **Limitación** | Dificultad con contexto de largo término |

---

> **Nota importante:** Las RNN son históricamente importantes en NLP pero han sido largementre superadas por **LSTM**, **GRU** y **Transformers**.

---

### 5.3 LSTM (Long Short-Term Memory Networks)

#### Idea Central

Las LSTMs introducen **mecanismos de gating** que regulan cómo la información es almacenada, olvidada y expuesta, permitiendo aprender sobre secuencias largas.

**Tres puertas (*gates*):**
1. **Input gate** - Qué información nueva almacenar
2. **Forget gate** - Qué información descartar
3. **Output gate** - Qué información exponer

**Referencias:**
- https://developer.nvidia.com/discover/lstm
- https://community.deeplearning.ai/t/week-4-visualizing-lstm-models/82053

| Aspecto | Detalle |
|---------|---------|
| **Implementación popular** | Capas LSTM en Keras y PyTorch |
| **Fortaleza** | Captura patrones temporales de largo plazo |

---

#### Ejemplo No-NLP: Análisis de Series Temporales Financieras

| Campo | Detalle |
|-------|---------|
| **Tarea** | Predecir tendencias de stock o volatilidad |
| **Por qué LSTM** | Captura patrones temporales de largo término |

---

#### Ejemplo NLP: Análisis de Sentimiento en Documentos Largos

| Campo | Detalle |
|-------|---------|
| **Tarea** | Clasificar sentimiento en reseñas de múltiples párrafos |
| **Por qué LSTM** | Mantiene contexto a través de muchas oraciones |
| **Framework** | Keras Embedding + LSTM |

---

> **Nota histórica:** Antes de los Transformers, las LSTMs fueron la arquitectura dominante en NLP para modelado secuencial.

**Analogía de Ingeniería:** LSTM es como un **sistema de memoria RAM con gestión inteligente**: tiene un controlador que decide qué datos cargar (input gate), qué datos liberar (forget gate) y qué datos enviar al CPU (output gate). Esto evita que la memoria se sature y permite retener información crítica por más tiempo.

---

### 5.4 GRU (Gated Recurrent Units)

#### Idea Central

Las GRUs simplifican las LSTMs combinando puertas, resultando en **menos parámetros** y entrenamiento más rápido mientras mantienen capacidad fuerte de modelado secuencial.

**Referencia:** https://www.geeksforgeeks.org/machine-learning/gated-recurrent-unit-networks/

| Diferencia con LSTM | Detalle |
|---------------------|---------|
| **No tiene** | Output gate |
| **Combina** | Input gate + Forget gate → Update gate |

| Aspecto | Detalle |
|---------|---------|
| **Implementación popular** | PyTorch nn.GRU |
| **Ventaja** | Eficiencia computacional |

---

#### Ejemplo No-NLP: Reconocimiento de Habla en Tiempo Real

| Campo | Detalle |
|-------|---------|
| **Contexto** | Sistemas embebidos o de baja latencia |
| **Ventaja** | Eficiencia computacional |

---

#### Ejemplo NLP: Traducción Automática Neuronal (Modelos Tempranos)

| Campo | Detalle |
|-------|---------|
| **Tarea** | Traducir oraciones entre idiomas |
| **Por qué GRU** | Rendimiento fuerte con menos recursos que LSTM |
| **Usado en** | Sistemas encoder-decoder seq2seq tempranos |

---

> **Nota:** Las GRUs son especialmente atractivas cuando la eficiencia del modelo es crítica.

---

### 5.5 Autoencoders

#### Idea Central

Los Autoencoders aprenden representaciones compactas y significativas de datos **reconstruyendo inputs** a través de una capa *bottleneck*.

**Referencia:** https://medium.com/data-science/applied-deep-learning-part-3-autoencoders-1c083af4d798

| Uso Típico | Descripción |
|------------|-------------|
| **Compresión** | Reducir dimensionalidad |
| **Denoising** | Eliminar ruido de datos |
| **Detección de anomalías** | Alto error de reconstrucción → anomalía |

| Aspecto | Detalle |
|---------|---------|
| **Implementación popular** | Deep Autoencoders en TensorFlow/Keras |

---

#### Ejemplo No-NLP: Detección de Fraude

| Campo | Detalle |
|-------|---------|
| **Enfoque** | Entrenar en transacciones normales |
| **Señal** | Alto error de reconstrucción → anomalía |

---

#### Ejemplo NLP: Sentence Embeddings y Reducción de Dimensionalidad

| Campo | Detalle |
|-------|---------|
| **Tarea** | Comprimir vectores TF-IDF o embeddings |
| **Caso de uso** | Clustering o similitud semántica |
| **Variante** | Autoencoders sequence-to-sequence usando LSTM |

---

> **Utilidad:** Los Autoencoders son útiles en NLP siempre que las etiquetas sean escasas o se necesiten aprender representaciones no supervisadas.

---

### 5.6 GNN (Graph Neural Networks)

#### Idea Central

Las GNNs operan sobre grafos, aprendiendo representaciones que combinan features de nodos con estructura relacional.

**Referencia:** https://towardsdatascience.com/practical-graph-neural-networks-for-molecular-machine-learning-5e6dee7dc003/

| Aspecto | Detalle |
|---------|---------|
| **Datos comunes** | Redes sociales, moléculas, knowledge graphs |
| **Implementación popular** | Graph Convolutional Networks (GCN) con PyTorch Geometric |

---

#### Ejemplo No-NLP: Predicción de Propiedades Moleculares

| Campo | Detalle |
|-------|---------|
| **Tarea** | Predecir toxicidad o solubilidad |
| **Por qué GNNs** | Las moléculas son grafos |

---

#### Ejemplo NLP: Grafos Semánticos y NLP Basado en Conocimiento

| Campo | Detalle |
|-------|---------|
| **Tarea** | Extracción de relaciones o question answering |
| **Enfoque** | Aplicar GNNs a árboles de dependencia o knowledge graphs |
| **Ejemplo** | Entity linking usando propagación en grafos |

---

> **Valor:** Las GNNs puentean la estructura simbólica y el aprendizaje neuronal en NLP.

---

### 5.7 GAN (Generative Adversarial Networks)

#### Idea Central

Las GANs consisten en dos redes competidoras:
- **Generator:** Produce datos sintéticos
- **Discriminator:** Distingue real de falso

**Entrenamiento:** Juego *minimax* entre los dos modelos.

**Referencia:** https://medium.com/sigmoid/a-brief-introduction-to-gans-and-how-to-code-them-2620ee465c30

| Aspecto | Detalle |
|---------|---------|
| **Implementación popular** | DCGAN y variantes en PyTorch |

---

#### Ejemplo No-NLP: Generación de Imágenes

| Campo | Detalle |
|-------|---------|
| **Tarea** | Generar caras realistas o arte |
| **Impacto** | Datos sintéticos de alta calidad y aplicaciones creativas |

---

#### Ejemplo NLP: Aumentación de Datos de Texto y Entrenamiento Adversarial

| Campo | Detalle |
|-------|---------|
| **Tarea** | Generar paráfrasis o oraciones sintéticas |
| **Desafío** | Los tokens discretos de texto dificultan el entrenamiento de GANs |
| **Soluciones** | SeqGAN, GANs basados en Reinforcement Learning |

---

> **Nota:** Las GANs son menos comunes en NLP que en visión, pero son valiosas para aumentación de datos y testing de robustez.

---

## 6. Contexto Adicional para Ingeniería de Software

### 6.1 Pipeline Típico de un Modelo NLP con Redes Neuronales

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Texto     │ →  │  Embedding   │ →  │   Capas     │ →  │   Salida     │
│   Raw       │    │   Layer      │    │   Neuronales│    │  (Softmax)   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

---

### 6.2 Consideraciones de Despliegue

| Arquitectura | Latencia | Memoria | Escalabilidad |
|--------------|----------|---------|---------------|
| **CNN** | Baja | Media | Alta |
| **RNN** | Media | Media | Media |
| **LSTM** | Alta | Alta | Media |
| **GRU** | Media | Media | Alta |
| **Transformer** | Media-Alta | Alta | Alta (con optimización) |

---

### 6.3 Regularización (Para Evitar Overfitting)

| Técnica | Descripción |
|---------|-------------|
| **Dropout** | Apagar neuronas aleatoriamente durante entrenamiento |
| **Weight decay** | Penalizar pesos grandes (L2 regularization) |
| **Early stopping** | Detener entrenamiento cuando validación deja de mejorar |
| **Batch normalization** | Normalizar activaciones por lote |

---

### 6.4 Comparativa: Convolución en Visión vs NLP

| Aspecto | Visión por Computadora | NLP |
|---------|------------------------|-----|
| **Input** | Imagen 2D (H × W × Canales) | Texto 1D (Secuencia × Dim_embedding) |
| **Filtro** | Kernel 2D (ej. 3×3) | Kernel 1D (ej. ventana de 3-5 palabras) |
| **Patrón aprendido** | Bordes, texturas, formas | N-grams, frases locales |
| **Invarianza deseada** | Translacional espacial | Posicional en secuencia |

---

## 7. Glosario de Términos Clave

| Término | Definición |
|---------|------------|
| **Backpropagation** | Algoritmo que calcula gradientes de la pérdida respecto a cada peso usando la regla de la cadena, permitiendo el ajuste de parámetros durante el entrenamiento. |
| **Embedding** | Representación vectorial densa y continua de palabras u otras unidades lingüísticas que captura similitud semántica. |
| **Logits** | Puntuaciones brutas de salida de una red neuronal antes de aplicar una función de activación como softmax. |
| **Vanishing Gradient** | Problema donde los gradientes se vuelven extremadamente pequeños al propagarse hacia atrás en redes profundas, dificultando el aprendizaje. |
| **One-Hot Encoding** | Representación vectorial donde solo un elemento es 1 (la clase verdadera) y todos los demás son 0. |

---

## 8. Preguntas de Autoevaluación

<details>
<summary><strong>❓ Pregunta 1:</strong> ¿Por qué ReLU es preferida sobre Sigmoid en capas ocultas?</summary>

**Respuesta:** ReLU es computacionalmente más barata (solo requiere una operación max), evita el problema de *vanishing gradients* (los gradientes no se desvanecen para valores positivos, siempre son 1), y introduce no-linealidad efectiva. Sigmoid sufre de gradientes muy pequeños en los extremos (cuando x es muy positivo o muy negativo), lo que ralentiza el entrenamiento en redes profundas porque los gradientes casi desaparecen al propagarse hacia atrás.
</details>

<details>
<summary><strong>❓ Pregunta 2:</strong> ¿Cuál es la diferencia principal entre LSTM y GRU?</summary>

**Respuesta:** LSTM tiene tres puertas (input gate, forget gate, output gate) mientras que GRU combina input y forget en una sola *update gate* y no tiene output gate. GRU tiene menos parámetros (aproximadamente 3/4 de los parámetros de LSTM), es más rápido de entrenar y requiere menos memoria, pero puede tener ligeramente menor capacidad de modelado que LSTM en tareas que requieren retener información muy a largo plazo.
</details>

<details>
<summary><strong>❓ Pregunta 3:</strong> ¿Por qué Softmax se combina típicamente con Cross-Entropy Loss?</summary>

**Respuesta:** Porque matemáticamente se complementan de forma elegante: cuando se usa softmax con cross-entropy, el gradiente se simplifica dramáticamente a (ŷ - y), donde ŷ es la predicción y y es la etiqueta verdadera. Esto hace el entrenamiento más eficiente computacionalmente y numéricamente estable, evitando cálculos complejos de derivadas.
</details>

<details>
<summary><strong>❓ Pregunta 4:</strong> ¿En qué escenario elegirías una CNN sobre una LSTM para NLP?</summary>

**Respuesta:** Elegiría CNN cuando: (1) el contexto local (n-grams) sea más importante que las dependencias de largo alcance, (2) la eficiencia computacional sea crítica (CNNs son más paralelizables), (3) se trabaje con textos cortos donde el contexto global no es esencial, o (4) se necesite baja latencia en inferencia. Ejemplos: clasificación de tweets, detección de spam, análisis de sentimiento de reseñas cortas. LSTM sería mejor para documentos largos, traducción automática, o generación de texto donde el contexto secuencial extenso importa.
</details>

<details>
<summary><strong>❓ Pregunta 5:</strong> ¿Qué papel juega el "bottleneck layer" en un Autoencoder?</summary>

**Respuesta:** El bottleneck layer (capa cuello de botella) fuerza a la red a aprender una representación comprimida y significativa de los datos, ya que debe transmitir toda la información esencial a través de una capa con menos neuronas que la entrada/salida. Esta restricción de dimensionalidad obliga al modelo a aprender las características más importantes y descartar ruido, permitiendo reducción de dimensionalidad, aprendizaje no supervisado de features, y detección de anomalías (datos que no se comprimen bien son atípicos).
</details>

<details>
<summary><strong>❓ Pregunta 6:</strong> ¿Qué es el problema de "Dead ReLUs" y cómo se puede mitigar?</summary>

**Respuesta:** El problema de *Dead ReLUs* ocurre cuando las neuronas se quedan atascadas outputeando 0 porque los pesos empujan constantemente los inputs a valores negativos. Una vez que una neurona está "muerta", su gradiente es 0 y no puede aprender. Mitigaciones: (1) Usar *Leaky ReLU* o *Parametric ReLU* que permiten pequeños gradientes negativos, (2) Inicialización cuidadosa de pesos (He initialization), (3) Learning rate más bajo, (4) Batch normalization para mantener activaciones en rango saludable.
</details>

---

## 9. Conclusión

Las redes neuronales han revolucionado el Procesamiento de Lenguaje Natural al permitir **aprendizaje automático de características** sin ingeniería manual exhaustiva. 

### Puntos Clave para Recordar:

1. **Funciones de activación** son cruciales: ReLU para capas ocultas, Softmax para salida multiclase
2. **Backpropagation** es el mecanismo eficiente para calcular gradientes y ajustar pesos
3. **Cada arquitectura tiene su nicho:**
   - CNN → patrones locales
   - RNN/LSTM/GRU → secuencias y contexto temporal
   - Autoencoder → representaciones no supervisadas
   - GNN → datos estructurados como grafos
   - GAN → generación de datos

4. **La elección de arquitectura depende de:**
   - Estructura de los datos (secuencial, gráfica, local)
   - Recursos disponibles (tiempo, memoria, datos)
   - Requisitos de la tarea (latencia, precisión, interpretabilidad)

5. **El Transformer** se ha convertido en la arquitectura dominante para modelos state-of-the-art, pero entender estas arquitecturas fundamentales es esencial para diseñar sistemas eficientes y apropiados para cada caso de uso.

---

## 10. Recursos Adicionales

| Tema | Enlace |
|------|--------|
| ¿Qué es una red neuronal? | https://mriquestions.com/what-is-a-neural-network.html |
| Convoluciones | https://medium.com/@bdhuma/6-basic-things-to-know-about-convolution-daef5e1bc411 |
| CNN en profundidad | https://developersbreach.com/convolution-neural-network-deep-learning/ |
| RNN vs FFNN | https://www.geeksforgeeks.org/data-analysis/difference-between-feed-forward-neural-networks-and-recurrent-neuralnetworks/ |
| LSTM | https://developer.nvidia.com/discover/lstm |
| GRU | https://www.geeksforgeeks.org/machine-learning/gated-recurrent-unit-networks/ |
| Autoencoders | https://medium.com/data-science/applied-deep-learning-part-3-autoencoders-1c083af4d798 |
| GNNs | https://towardsdatascience.com/practical-graph-neural-networks-for-molecular-machine-learning-5e6dee7dc003/ |
| GANs | https://medium.com/sigmoid/a-brief-introduction-to-gans-and-how-to-code-them-2620ee465c30 |

---

*Documento generado para fines educativos - Guía de estudio basada en "Neural networks for NLP.pdf"*
