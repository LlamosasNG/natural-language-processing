# Marco Teórico: Clasificación de Texto mediante Perceptrón Multicapa (MLP)

---

## 1. Homogeneización y Flujo de Secuencias en Procesamiento de Lenguaje Natural

En el Procesamiento de Lenguaje Natural (PLN) basado en arquitecturas de aprendizaje profundo conexionistas de tipo *Feedforward*, uno de los principales retos radica en la naturaleza variable del lenguaje. Las oraciones y documentos presentan longitudes diversas, mientras que los modelos de Perceptrón Multicapa (MLP) tradicionales requieren vectores de entrada de dimensionalidad estática y homogénea.

### 1.1. Truncamiento y Relleno (*Padding*) de Secuencias
Para unificar la dimensionalidad de las entradas, es necesario aplicar transformaciones que lleven secuencias discretas arbitrarias a una longitud estándar fija $L$.

*   **Truncamiento:** Proceso mediante el cual una secuencia de longitud $n > L$ es recortada para retener únicamente los primeros $L$ elementos. Desde una perspectiva lingüística, este proceso asume que la mayor carga semántica y la intención temática del documento se concentran en su sección inicial.
*   **Relleno (*Padding*):** Proceso aplicado a secuencias de longitud $n < L$, en el cual se añaden elementos nulos o tokens de relleno (denominados lógicamente como `<PAD>`) en posiciones predefinidas (usualmente al final de la secuencia, conocido como *post-padding*) hasta completar la longitud estándar $L$.

Matemáticamente, dada una secuencia discreta de tokens $S = [s_1, s_2, \dots, s_n]$, la secuencia homogeneizada $\bar{S}$ de tamaño fijo $L$ se define mediante la función de mapeo:

$$\bar{S} = \begin{cases} 
[s_1, s_2, \dots, s_L] & \text{si } n \geq L \\
[s_1, s_2, \dots, s_n, \underbrace{\text{PAD}, \text{PAD}, \dots, \text{PAD}}_{L - n}] & \text{si } n < L
\end{cases}$$

---

### 1.2. Procesamiento por Mini-lotes (*Mini-batch Processing*) y Barajado
En el entrenamiento de redes neuronales, el ajuste de parámetros a través del gradiente de la función de pérdida sobre el dataset completo (método de lote completo o *Batch*) es computacionalmente inviable en grandes corpus debido a restricciones de memoria física. En el extremo opuesto, el método estocástico puro (muestra por muestra) induce una alta varianza y desaprovecha las capacidades de computación paralela en hardware especializado (GPUs/TPUs).

El **Procesamiento por Mini-lotes** equilibra ambos enfoques al agrupar el conjunto de datos de entrenamiento en subconjuntos de tamaño $B$ (donde $B \in \mathbb{N}$). La estimación del gradiente se calcula como el promedio de los gradientes de las muestras pertenecientes a dicho subconjunto:

$$\mathbf{g} \approx \frac{1}{B} \sum_{i=1}^{B} \nabla_\theta \mathcal{L}_i(\theta)$$

#### Importancia del Barajado Aleatorio (*Shuffling*)
El barajado aleatorio de las secuencias al inicio de cada época de entrenamiento es un requisito teórico indispensable en el aprendizaje supervisado. Este proceso garantiza que las muestras contenidas en cada mini-lote sean independientes e idénticamente distribuidas (i.i.d.). Sin este mecanismo, el optimizador podría sesgarse hacia patrones secuenciales o correlaciones temporales espurias presentes en el orden físico de almacenamiento de los datos de entrada.

---

## 2. Redes Neuronales Feedforward (FFNN) y Mecanismos de Pooling

Una **Red Neuronal Feedforward (FFNN)** o **Perceptrón Multicapa (MLP)** es una arquitectura en la que la información se propaga en un solo sentido, desde la capa de entrada hasta la de salida, a través de capas intermedias (ocultas) formadas por nodos de procesamiento interconectados.

### 2.1. Agregación Temporal: *Global Average Pooling* con Enmascaramiento
En tareas de clasificación de texto, tras la fase de codificación densa (representación mediante embeddings), cada documento se expresa como una secuencia de vectores densos $\mathbf{E} = [\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_L]$ en un espacio de dimensionalidad $D$, conformando una matriz de dimensiones $L \times D$.

Para alimentar un clasificador lineal, esta matriz debe colapsarse en un único vector representativo $\mathbf{x} \in \mathbb{R}^D$. El método matemático óptimo es el **Promedio Global (*Global Average Pooling*)**. Sin embargo, la inclusión de tokens de relleno en secuencias con *padding* introduce ruido semántico nulo que reduce artificialmente la magnitud media del vector.

Para resolver esta limitación, se define conceptualmente un **mecanismo de enmascaramiento booleano**:

1.  **Definición de la función indicadora:** Se genera una variable indicadora $M_t \in \{0, 1\}$ para cada posición de la secuencia temporal $t$:
    $$M_t = \begin{cases} 
    1 & \text{si el token en } t \text{ es un elemento real} \\
    0 & \text{si el token en } t \text{ es de relleno (PAD)} 
    \end{cases}$$
2.  **Suma enmascarada:** Se descartan las contribuciones semánticas de las posiciones nulas sumando exclusivamente los vectores reales:
    $$\mathbf{s} = \sum_{t=1}^{L} \mathbf{e}_t \cdot M_t$$
3.  **Normalización por longitud real:** Se calcula el promedio dividiendo la suma únicamente por la cardinalidad del conjunto de tokens válidos, evitando la indeterminación matemática mediante una función de acotación inferior:
    $$\ell = \max\left(1, \sum_{t=1}^{L} M_t\right)$$
4.  **Vector consolidado resultante:**
    $$\mathbf{x} = \frac{\mathbf{s}}{\ell}$$

---

### 2.2. Capas Totalmente Conectadas (*Linear Layers*)
La estructura fundamental del MLP se compone de transformaciones afines sucesivas. Una capa totalmente conectada o lineal realiza una proyección matemática que mapea un espacio de entrada de dimensión $N_{\text{in}}$ a uno de salida de dimensión $N_{\text{out}}$:

$$\mathbf{z} = \mathbf{W} \mathbf{a} + \mathbf{b}$$

Donde:
*   $\mathbf{a} \in \mathbb{R}^{N_{\text{in}}}$ representa el vector de activaciones o características de la capa previa.
*   $\mathbf{W} \in \mathbb{R}^{N_{\text{out}} \times N_{\text{in}}}$ es la matriz de pesos (*weights*), cuyos elementos cuantifican la fuerza de la conexión sináptica entre nodos.
*   $\mathbf{b} \in \mathbb{R}^{N_{\text{out}}}$ es el vector de sesgo (*bias*), que permite desplazar la frontera de decisión fuera del origen coordenado.
*   $\mathbf{z} \in \mathbb{R}^{N_{\text{out}}}$ es el vector pre-activación resultante.

---

## 3. No Linealidad y Regularización en Capas Ocultas

La composición de múltiples capas puramente lineales se reduce algebraicamente a una única transformación lineal. Por lo tanto, para dotar al modelo de la capacidad de aproximar fronteras de decisión altamente complejas y no lineales (garantizado por el Teorema de Aproximación Universal), es imperativo aplicar funciones de activación no lineales.

### 3.1. Función de Activación ReLU (*Rectified Linear Unit*)
La unidad lineal rectificada (ReLU) es una función de activación no lineal y continua por partes que actúa como un rectificador de señal. Su definición formal es:

$$f(z) = \max(0, z)$$

Su subgradiente se define como:

$$f'(z) = \begin{cases} 
1 & \text{si } z > 0 \\
0 & \text{si } z < 0 
\end{cases}$$

```
   f(z) ^
        |        /
        |       /
        |      /
        |     /
  ------|----/-------> z
        |
```

#### Ventajas Teóricas en Deep Learning
*   **Eficiencia Computacional Extrema:** Prescinde de operaciones trascendentes complejas (como exponenciales en funciones sigmoides), requiriendo únicamente una comparación a nivel de registros de procesador.
*   **Mitigación del Desvanecimiento del Gradiente (*Vanishing Gradient*):** En la región de valores positivos ($z > 0$), la derivada de la función es constantemente $1.0$. Esto permite que el flujo de error se transmita intacto hacia atrás durante la retropropagación.
*   **Activación Dispersa (*Sparsity*):** Al mapear a $0$ todas las entradas negativas, la red induce una representación dispersa en la cual solo un subconjunto de neuronas se activa ante determinados estímulos, emulando la eficiencia metabólica del cerebro biológico.

#### Saturación por ReLU Muerta (*Dying ReLU*)
Una limitación teórica de la función ReLU ocurre cuando una tasa de aprendizaje excesiva o una mala inicialización de pesos desplaza los parámetros de modo que una neurona reciba entradas negativas en todo el corpus. En este estado, tanto la activación como su derivada son cero, impidiendo que el gradiente fluya y provocando que la neurona quede inactiva permanentemente.

---

### 3.2. Regularización por Descarte Neuronal (*Dropout*)
En redes con una cantidad significativa de parámetros libres, el sobreajuste (*overfitting*) es una vulnerabilidad crítica donde el modelo memoriza el ruido estadístico de los datos de entrenamiento. El **Dropout** aborda esto desde una perspectiva probabilística de regularización estructural.

Durante el entrenamiento, para cada mini-lote, el Dropout desactiva de forma aleatoria e independiente cada neurona de una capa con una probabilidad de descarte $p \in (0, 1)$ (siendo común $p = 0.3$).

Matemáticamente, la activación de la capa $\mathbf{a}$ se perturba mediante un producto Hadamard con un vector de máscara estocástica $\mathbf{r}$:

$$\mathbf{a}_{\text{dropout}} = \mathbf{a} \odot \mathbf{r}$$

Donde los componentes del vector $\mathbf{r}$ se extraen de una distribución de Bernoulli:

$$r_j \sim \text{Bernoulli}(1 - p)$$

```
     Capa Oculta Estándar                    Capa Oculta con Dropout (p=0.3)
         ( )   ( )   ( )                          ( )   (x)   ( )
        /   \ /   \ /   \                        /       /     \
       ( )   ( )   ( )                          (x)     ( )     (x)
```

#### Fundamento Teórico de su Efectividad
1.  **Prevención de Coadaptación Semántica:** Al verse privada aleatoriamente de la contribución de neuronas vecinas, cada neurona debe aprender características robustas e independientes, impidiendo que dependa de configuraciones ultra-específicas de otros nodos.
2.  **Ensamble Estocástico de Modelos:** Dropout puede interpretarse como un método para entrenar de forma simultánea un ensamble masivo de $2^N$ sub-arquitecturas de redes neuronales (donde $N$ es el número de neuronas) que comparten pesos. Esto reduce drásticamente la varianza estadística del clasificador.
3.  **Escalamiento en Fase de Inferencia:** Durante la evaluación del modelo (*inference*), el dropout se desactiva. Todas las neuronas permanecen activas, pero sus valores de activación deben escalarse multiplicándolos por $(1-p)$ para que el valor esperado de las activaciones sea equivalente al del entrenamiento.

---

## 4. Optimización, Pérdida y Retropropagación en Clasificación Multiclase

La optimización de una red neuronal consiste en la búsqueda de un conjunto de parámetros $\theta = \{\mathbf{W}, \mathbf{b}\}$ que minimice de forma iterativa el error predictivo sobre un corpus etiquetado por expertos humanos (*Gold Labels*).

### 4.1. Clasificación Multiclase y la Función *Softmax*
En un problema de clasificación multiclase con $K$ categorías, la capa de salida del MLP produce un vector de puntuaciones continuas y no acotadas denominadas *logits*, $\mathbf{z} \in \mathbb{R}^K$.

Para mapear este vector en una distribución discreta de probabilidad que represente el grado de certeza del modelo sobre cada categoría, se aplica la función de normalización exponencial **Softmax** sobre cada puntuación $z_i$:

$$\hat{y}_i = \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

#### Propiedades Matemáticas de la Función Softmax
*   **Acotación Probabilística:** $\forall i \in \{1, \dots, K\}, \quad 0 < \hat{y}_i < 1$.
*   **Consistencia de Probabilidad Total:** $\sum_{i=1}^{K} \hat{y}_i = 1.0$.
*   **Comportamiento de Maximizador Suave:** Amplifica de forma no lineal los componentes de mayor valor relativo en $\mathbf{z}$ y atenúa los menores, ofreciendo una aproximación diferenciable de la función `arg max`.

---

### 4.2. Función de Pérdida de Entropía Cruzada Multiclase (*Categorical Cross-Entropy Loss*)
Derivada de la teoría de la información de Shannon, la entropía cruzada mide la discrepancia entre la distribución real de probabilidad $\mathbf{y}$ (etiqueta verdadera codificada en *One-Hot*) y la distribución estimada por la red $\hat{\mathbf{y}}$.

Su formulación general es:

$$\mathcal{L}_{\text{CE}} = - \sum_{i=1}^{K} y_i \log(\hat{y}_i)$$

Siendo $\mathbf{y}$ un vector *One-Hot* donde la única clase real $c$ posee valor $y_c = 1$, la pérdida se reduce a la verosimilitud logarítmica negativa de la probabilidad estimada para la clase correcta:

$$\mathcal{L}_{\text{CE}} = - \log(\hat{y}_c)$$

#### Consideraciones de Estabilidad Numérica: *Log-Sum-Exp*
La evaluación directa de la exponencial en la ecuación de Softmax puede inducir problemas de desbordamiento de memoria por arriba (*overflow*, cuando $z_i \gg 0$) o por abajo (*underflow*, cuando $z_i \ll 0$). Para evitar la pérdida de precisión de punto flotante en sistemas computacionales, las funciones de pérdida modernas integran la softmax y la entropía cruzada en un solo paso matemático estable utilizando la técnica *Log-Sum-Exp*, restando el logit de máxima magnitud ($z_{\max}$) antes de la exponenciación:

$$\log\left(\sum_{j=1}^{K} e^{z_j}\right) = z_{\max} + \log\left(\sum_{j=1}^{K} e^{z_j - z_{\max}}\right)$$

---

### 4.3. Algoritmo de Optimización *Adam (Adaptive Moment Estimation)*
El optimizador es el operador matemático que calcula las reglas de actualización de los parámetros con el fin de converger hacia un mínimo global o local de la pérdida. El algoritmo **Adam** es un método de descenso de gradiente estocástico de tasa de aprendizaje adaptativa.

Adam estima de forma dinámica el promedio de los gradientes (momento lineal o sesgo del gradiente) y la varianza no centrada de los gradientes (escala del gradiente) para adaptar la tasa de aprendizaje de cada parámetro individual:

1.  **Estimación del primer momento ($1^{\text{er}}$ momento de los gradientes):**
    $$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t$$
2.  **Estimación del segundo momento (momento no centrado de gradientes al cuadrado):**
    $$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$$
3.  **Corrección analítica del sesgo de inicialización:** Dado que $\mathbf{m}_0$ y $\mathbf{v}_0$ se inicializan en cero, las estimaciones iniciales están sesgadas hacia dicho valor. La corrección para cada paso de tiempo $t$ se define como:
    $$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$
4.  **Actualización de parámetros:**
    $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \hat{\mathbf{m}}_t$$

Donde:
*   $\eta$ representa la tasa de aprendizaje global (*learning rate*).
*   $\beta_1$ y $\beta_2$ son los hiperparámetros de decaimiento exponencial para la estimación de momentos (típicamente configurados como $\beta_1 = 0.9$ y $\beta_2 = 0.999$).
*   $\epsilon$ es un término de estabilidad numérica para evitar la división entre cero (usualmente $\epsilon = 1\text{e-}8$).

---

### 4.4. Retropropagación del Error (*Backpropagation*) y Autodiferenciación
El cálculo del gradiente de la función de pérdida respecto a la totalidad de parámetros del modelo se sustenta en el algoritmo de **Retropropagación (*Backpropagation*)**. Este método aplica la **regla de la cadena multivariada** del cálculo diferencial, permitiendo el flujo de derivadas parciales en sentido opuesto a la propagación hacia adelante.

Matemáticamente, para una capa lineal arbitraria $l$, el gradiente del costo respecto a los pesos $\mathbf{W}^{[l]}$ y sesgos $\mathbf{b}^{[l]}$ se expresa como:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}} \left(\mathbf{a}^{[l-1]}\right)^T$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}}$$

El error propagado pre-activación $\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}}$ se transmite recursivamente hacia las capas inferiores utilizando la matriz transpuesta de pesos de la capa superior y la derivada de la función de activación correspondiente ($\sigma'$):

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}} = \left( \left(\mathbf{W}^{[l+1]}\right)^T \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l+1]}} \right) \odot \sigma'\left(\mathbf{z}^{[l]}\right)$$

#### Grafos de Computación Dinámica
En el cómputo moderno, este proceso se gestiona mediante un motor de **Autodiferenciación** que construye de manera dinámica un Grafo Acíclico Dirigido (DAG) durante el recorrido de ida (*Forward Pass*). Las hojas de dicho grafo representan las variables que requieren el cómputo del gradiente ($\nabla$). Las variables cuyos gradientes están inhabilitados quedan completamente fuera de la cadena de derivadas, lo que optimiza significativamente el costo en tiempo de cómputo y el uso de memoria RAM durante la fase de optimización.
