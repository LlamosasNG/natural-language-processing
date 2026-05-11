# Marco Teórico

## 1. Clasificación de Texto

La clasificación de texto es una tarea fundamental dentro del Procesamiento de Lenguaje Natural (PLN) que consiste en asignar automáticamente una categoría predefinida a un documento de texto. En el ámbito del aprendizaje automático supervisado, esta tarea se aborda mediante modelos que aprenden a mapear una representación vectorial del texto a una etiqueta de clase, utilizando para ello un conjunto de ejemplos previamente etiquetados por expertos humanos, conocidos como *gold labels* o etiquetas de oro.

Entre las aplicaciones más comunes de la clasificación de texto se encuentran: el análisis de sentimientos en reseñas de productos o servicios, la detección de spam en correos electrónicos, la categorización temática de noticias y la identificación de lenguaje ofensivo. En todos los casos, el objetivo compartido es construir un modelo capaz de generalizar patrones textuales para clasificar documentos no vistos durante el entrenamiento.

## 2. Extracción de Características para Representación Textual

### 2.1 El Problema de la Vectorización

Los modelos de aprendizaje automático, por su naturaleza matemática, requieren que la información de entrada se encuentre en formato numérico. El texto, siendo una secuencia de símbolos categóricos (palabras, caracteres), no puede ser procesado directamente por estos algoritmos. Por esta razón, resulta necesario aplicar un proceso de transformación que convierta cada documento en un vector de características numéricas: un proceso denominado *feature extraction* o extracción de características.

Este proceso es determinante para el rendimiento del clasificador, ya que la calidad de las características extraídas limita directamente la capacidad del modelo para distinguir entre clases. Una representación deficiente impedirá que incluso el algoritmo de clasificación más sofisticado pueda aprender los patrones relevantes.

### 2.2 Métodos de Vectorización

Existen diversos enfoques para representar texto como vectores numéricos. Entre los más utilizados se encuentran:

- **Bag of Words (BoW):** Este método cuenta la frecuencia de aparición de cada palabra en el documento, ignorando el orden y la estructura gramatical. Cada documento se representa como un vector donde cada dimensión corresponde a una palabra del vocabulario y el valor es su frecuencia en dicho documento.

- **TF-IDF (*Term Frequency - Inverse Document Frequency*):** A diferencia del conteo simple, TF-IDF pondera las frecuencias de las palabras considerando su importancia relativa dentro del corpus. Las palabras que aparecen frecuentemente en un documento pero raramente en el resto del corpus reciben un peso mayor, mientras que las palabras comunes (como artículos o preposiciones) reciben un peso menor. Esto permite destacar términos discriminativos.

- **Word Embeddings:** Los embeddings de palabras representan cada término como un vector denso en un espacio de baja dimensionalidad, donde palabras con significados similares se ubicан próximas entre sí. Modelos como Word2Vec o GloVe aprenden estas representaciones a partir de grandes corpus textuales.

### 2.3 Extracción Manual de Características

En determinados contextos, particularmente cuando se trabaja con dominios específicos o conjuntos de datos pequeños, resulta conveniente definir manualmente las características relevantes. Este enfoque consiste en diseñar *lexicones* o conjuntos de palabras clave que correspondan a conceptos Semanticamente relevantes para la tarea de clasificación.

Por ejemplo, en un problema de análisis de sentimientos sobre reseñas de películas, un lexicón positivo podría contener palabras como *"excellent"*, *"amazing"*, *"wonderful"*, mientras que uno negativo incluiría términos como *"terrible"*, *"boring"*, *"awful"*. Cada documento se transforma entonces en un vector donde cada dimensión representa el conteo de palabras presentes en el lexicón correspondiente.

### 2.4 Función Sigmoide

Una vez obtenido el vector de características $x = [x_1, x_2, ..., x_n]$, el modelo de clasificación calcula una combinación lineal de dichas características:

$$z = w \cdot x + b = \sum_{i=1}^{n} w_i x_i + b$$

donde $w$ son los pesos aprendidos durante el entrenamiento y $b$ es el sesgo (*bias*). El resultado $z$ puede ser cualquier número real, lo cual dificulta su interpretación directa como probabilidad de pertenencia a una clase.

La **función sigmoide** resuelve este problema al transformar cualquier valor de $z$ a un rango entre 0 y 1:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Esta función tiene forma de "S" suave y presenta las siguientes propiedades:
- Cuando $z \to +\infty$, $\sigma(z) \to 1$
- Cuando $z \to -\infty$, $\sigma(z) \to 0$
- Cuando $z = 0$, $\sigma(z) = 0.5$

De este modo, la salida de la sigmoide puede interpretarse como la **probabilidad** de que el documento pertenezca a la clase positiva. Si $\hat{y} = \sigma(z) \geq 0.5$, el modelo clasifica el documento como positivo; en caso contrario, como negativo.

## 3. Regresión Logística

### 3.1 Definición

El clasificador de regresión logística es un modelo lineal de clasificación binaria que, pese a su nombre, se utiliza para tareas de clasificación y no de regresión. Su funcionamiento se basa en calcular una suma ponderada de las características de entrada (la función $z$), aplicar la función sigmoide para obtener una probabilidad, y finalmente tomar una decisión en función de un umbral, típicamente 0.5.

### 3.2 Función de Pérdida: Entropía Cruzada

Para entrenar el modelo, es necesario cuantificar qué tan lejos está la predicción $\hat{y}$ del valor real $y$. La **entropía cruzada** (*cross-entropy*) es la función de pérdida estándar para problemas de clasificación binaria:

$$L_{CE}(y, \hat{y}) = -\left( y \log \hat{y} + (1-y) \log(1-\hat{y}) \right)$$

Esta función presenta una propiedad fundamental: penaliza de manera más intensa las predicciones que el modelo hace con alta confianza pero que resultan incorrectas. Si el modelo predice $\hat{y} = 0.99$ (altamente seguro de clase positiva) pero el valor real es $y = 0$, la pérdida es aproximadamente 4.6. En contraste, si la predicción fuera $\hat{y} = 0.6$, la pérdida sería aproximadamente 0.51. Esta característica impulsa al modelo a ser cauto cuando no está seguro.

### 3.3 Descenso de Gradiente

El algoritmo de **descenso de gradiente** (*gradient descent*) es el método utilizado para ajustar los pesos del modelo con el objetivo de minimizar la función de pérdida. El proceso opera de forma iterativa:

1. Para cada ejemplo de entrenamiento, calcular la predicción $\hat{y}$
2. Calcular el error: $\text{error} = y - \hat{y}$
3. Actualizar los pesos: $\theta \leftarrow \theta + \eta \cdot \text{error} \cdot x_i$

Donde $\eta$ es la **tasa de aprendizaje** (*learning rate*), un hiperparámetro que controla el tamaño de los ajustes realizados en cada iteración. Una tasa muy alta puede causar que el algoritmo oscile o diverja; una tasa muy baja resulta en un aprendizaje excesivamente lento.

Tras múltiples pasadas por todos los ejemplos de entrenamiento (cada pasada se denomina *época*), los pesos convergen hacia valores que minimizan la pérdida promedio en el conjunto de entrenamiento.

## 4. División de Datos: Training, Development y Test Sets

### 4.1 La Necesidad de Separar Datos

Un principio fundamental en aprendizaje automático es que un modelo debe evaluarse con datos que no ha observado durante su entrenamiento. Si se entrena y evalúa con el mismo conjunto de datos, el modelo podría simplemente memorizar las respuestas en lugar de aprender patrones generalizables, fenómeno conocido como **sobreajuste** (*overfitting*).

Por esta razón, resulta imprescindible dividir el corpus disponible en múltiples subconjuntos, cada uno con una función específica en el proceso de desarrollo del modelo.

### 4.2 Los Tres Conjuntos

- **Training Set:** Es el conjunto de datos utilizado para ajustar los parámetros del modelo (los pesos $\theta$ en el caso de la regresión logística). El modelo aprende directamente de estos datos, por lo que debe constituir la mayor proporción del corpus.

- **Development Set (Dev Set):** Es un conjunto de datos mantenido separado del entrenamiento, utilizado para ajustar hiperparámetros (tasa de aprendizaje, número de épocas), comparar diferentes versiones del modelo y detectar señales de sobreajuste. El modelo no aprende de este conjunto, solo observa su comportamiento para tomar decisiones de diseño.

- **Test Set:** Es el conjunto de datos reservado exclusivamente para la evaluación final del modelo. No debe utilizarse durante el desarrollo ni el ajuste de hiperparámetros, ya que constituye la única medida imparcial del rendimiento esperado del modelo en datos no vistos.

### 4.3 Proporciones Típicas

Las proporciones habituales para la división son:
- **Training Set:** 60% - 80%
- **Development Set:** 10% - 20%
- **Test Set:** 10% - 20%

La elección específica depende del tamaño total del corpus. Conjuntos más grandes permiten fracciones más pequeñas para Dev y Test sin perder poder estadístico en las evaluaciones.

### 4.4 Estratificación

Cuando las clases se encuentran desbalanceadas (por ejemplo, 90% negativos y 10% positivos), la división aleatoria simple podría distribuir las clases de manera desigual entre los conjuntos. La **estratificación** es una técnica que garantiza que cada subconjunto mantenga la misma proporción de cada clase que el corpus original, asegurando representatividad en todos los conjuntos.

### 4.5 Prevención de Fuga de Datos

La **fuga de datos** (*data leakage*) ocurre cuando información del conjunto de prueba se filtra inadvertidamente hacia el proceso de entrenamiento,inflando artificialmente las métricas de evaluación. Esta situación produce resultados engañosamente optimistas y un modelo que fallará en producción.

Un error frecuente consiste en ajustar los parámetros de preprocesamiento (como la media y desviación estándar para normalización, o el vocabulario del vectorizador TF-IDF) utilizando la totalidad del corpus antes de dividirlo. Los parámetros de preprocesamiento deben calcularse **únicamente** con datos del Training Set y aplicarse posteriormente a Dev y Test.

## 5. Evaluación de Clasificadores

### 5.1 La Matriz de Confusión

La matriz de confusión es una tabla que desglosa las predicciones del modelo según cuatro categorías fundamentales:

|  | Predicción Negativa | Predicción Positiva |
|---|---|---|
| **Real Negativo** | TN (Verdadero Negativo) | FP (Falso Positivo) |
| **Real Positivo** | FN (Falso Negativo) | TP (Verdadero Positivo) |

- **TN (*True Negative*):** El modelo predijo negativo y la respuesta real era negativa.
- **FP (*False Positive*):** El modelo predijo positivo, pero la respuesta real era negativa. También denominado Error Tipo I.
- **FN (*False Negative*):** El modelo predijo negativo, pero la respuesta real era positiva. También denominado Error Tipo II.
- **TP (*True Positive*):** El modelo predijo positivo y la respuesta real era positiva.

### 5.2 Métricas de Rendimiento

A partir de los valores de la matriz de confusión, se calculan diversas métricas:

**Accuracy (Exactitud):**
$$Acc = \frac{TP + TN}{TP + FP + TN + FN}$$

Representa el porcentaje total de predicciones correctas. Su limitación principal es que resulta engañosa cuando las clases están desbalanceadas: un clasificador que siempre predice la clase mayoritaria puede alcanzar accuracy muy alto sin haber aprendido patrones útiles.

**Precision (Precisión):**
$$Prec = \frac{TP}{TP + FP}$$

Indica, de todas las instancias clasificadas como positivas por el modelo, cuántas realmente lo eran. Responde a la pregunta: *"Cuando el modelo dice que es positivo, ¿qué tan confiable es?"*

**Recall (Exhaustividad):**
$$Rec = \frac{TP}{TP + FN}$$

Indica, de todas las instancias realmente positivas, cuántas fueron identificadas por el modelo. Responde a la pregunta: *"De todos los positivos reales, ¿cuántos capturó el modelo?"*

### 5.3 El Trade-off entre Precision y Recall

Existe una tensión inherente entre Precision y Recall. Modificar el umbral de decisión afecta simultáneamente ambas métricas:

- **Aumentar el umbral** (ej. solo clasificar como positivo si $\hat{y} > 0.9$): Incrementa Precision (el modelo es más exigente) pero reduce Recall (se pierden positivos borderline).
- **Reducir el umbral** (ej. clasificar como positivo si $\hat{y} > 0.3$): Incrementa Recall (captura más positivos) pero reduce Precision (más falsos positivos).

### 5.4 F1-Score

El **F1-Score** resuelve el trade-off entre Precision y Recall al calcular su media armónica:

$$F_1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$$

La media armónica penaliza más los valores desbalanceados que la media aritmética. Si una métrica es alta (ej. Precision = 1.0) pero la otra es baja (ej. Recall = 0.1), el F1-Score será bajo (aproximadamente 0.18), indicando que el modelo no tiene un rendimiento equilibrado.

### 5.5 Curva ROC y AUC

La curva **ROC** (*Receiver Operating Characteristic*) graphical el *True Positive Rate* (Recall) contra el *False Positive Rate* para todos los posibles umbrales de decisión:

$$TPR = \frac{TP}{TP + FN} \quad FPR = \frac{FP}{FP + TN}$$

Puntos relevantes de la curva:
- **Esquina superior izquierda:** Rendimiento ideal (Recall = 1, FPR = 0)
- **Línea diagonal:** Rendimiento equivalente a adivinar al azar
- **Por debajo de la diagonal:** Rendimiento peor que el azar

El **AUC** (*Area Under the Curve*) cuantifica el área bajo la curva ROC:
- **AUC = 1.0:** Separación perfecta entre clases
- **AUC = 0.5:** Sin poder discriminativo (equivalente al azar)
- **AUC < 0.5:** El modelo está invertido; invertir las predicciones mejoraría el rendimiento

El AUC tiene la ventaja de ser independiente del umbral de decisión, proporcionando una medida agregada de la capacidad del modelo para separar clases.

### 5.6 Selección de Métricas según el Contexto

La elección de la métrica principal depende de la aplicación específica:

| Escenario | Métrica Prioritaria | Razón |
|---|---|---|
| Clasificación general | F1-Score | Balance entre no perder positivos y no acumular falsos positivos |
| Filtros de spam | Precision | Priorizar no mover emails legítimos a spam (FP) sobre no detectar spam (FN) |
| Detección de enfermedades | Recall | Priorizar no enviar a casa a un enfermo (FN) sobre pruebas adicionales por falsos positivos (FP) |

## 6. Síntesis del Pipeline de Clasificación

El proceso completo de clasificación de texto mediante aprendizaje supervisado integra todos los conceptos descritos:

1. **Representación textual:** Los documentos se transforman en vectores de características numéricas mediante técnicas de vectorización.
2. **División de datos:** El corpus se separa en Training, Development y Test sets con estratificación cuando es necesario.
3. **Entrenamiento:** El modelo ajusta sus parámetros mediante descenso de gradiente, minimizando la entropía cruzada sobre el Training Set.
4. **Validación:** El Development Set permite afinar hiperparámetros y detectar sobreajuste.
5. **Evaluación final:** El Test Set proporciona una medición imparcial del rendimiento real del modelo, reportando métricas como Accuracy, Precision, Recall, F1-Score y AUC.

Este pipeline garantiza que el modelo resultante no solo haya aprendido patrones genuinos del dominio, sino que además sea capaz de generalizarlos a datos no vistos durante su desarrollo.