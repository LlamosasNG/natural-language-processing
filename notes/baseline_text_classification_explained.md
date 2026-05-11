# Explicación Detallada del Código: Baseline Text Classification

Este documento es una guía paso a paso del código de **Baseline Text Classification.ipynb**, haciendo referencia constante a los conceptos teóricos de los PDFanalizados.

---

## 1. Carga de Dependencias y Datos

### Código:
```python
!pip install -q datasets
from datasets import load_dataset, concatenate_datasets
```

### Explicación:
Se instalan e importan las herramientas necesarias para trabajar con datasets de Hugging Face. `load_dataset` permite descargar corpus estándar de NLP, y `concatenate_datasets` fusiona diferentes conjuntos de datos.

---

## 2. Carga del Dataset (IMDB)

### Código:
```python
imdb = load_dataset("imdb")
# imdb["train"] → 25,000 reviews
# imdb["test"]  → 25,000 reviews
```

### Explicación:
Se carga el dataset **IMDB**, un corpus clásico de clasificación de sentimientos que contiene 25,000 reseñas de películas para entrenamiento y 25,000 para prueba. Este es el dataset base sobre el cual se trabaja.

---

## 3. Unificación del Corpus

### Código:
```python
full_dataset = concatenate_datasets([
    imdb["train"],
    imdb["test"]
])
print("Total corpus size:", len(full_dataset))  # 50,000
```

### Explicación:
En lugar de entrenar directamente con las divisiones predefinidas (train/test), se unen ambos conjuntos. Esto es una **buena práctica** porque ahora tenemos control total sobre cómo dividir los datos, siguiendo las reglas del PDF de *Training and Test Sets* (evitar fugas, usar estratificación, crear Dev Set).

---

## 4. Primera División: Train y Temporal

### Código:
```python
split_1 = full_dataset.train_test_split(
    test_size=0.4,
    seed=42,
    shuffle=True
)
train_set = split_1["train"]      # 80%
temp_set  = split_1["test"]       # 40%
```

### Explicación:
Se divide el corpus unificado en:
*   **Train Set (80%):** El conjunto donde el modelo "estudiará" los patrones.
*   **Temp Set (40%):** Este conjunto aún no tiene nombre, pero más adelante se dividirá en Dev y Test.

**Parámetros importantes:**
*   `test_size=0.4`: Indica que el 40% del corpus completo irá al conjunto temporal.
*   `seed=42`: Garantiza que la división sea **reproducible**. Si alguien más corre el código con la misma semilla, obtendrá exactamente las mismas divisiones.
*   `shuffle=True`: **Mezcla los datos** antes de dividirlos. Esto es crucial porque los datos suelen venir ordenados (ej. todas las positivas primero), y sin mezclar, el modelo aprendería un patrón sesgado.

---

## 5. Segunda División: Dev y Test

### Código:
```python
split_2 = temp_set.train_test_split(
    test_size=0.5,
    seed=42,
    shuffle=True
)
dev_set  = split_2["train"]       # 20%
test_set = split_2["test"]        # 20%
```

### Explicación:
El conjunto temporal se divide en dos partes iguales:
*   **Dev Set (20%):** El "cuestionario de práctica". Se usa para ajustar hiperparámetros y detectar *overfitting*.
*   **Test Set (20%):** El "examen final". **Solo se toca una vez** al final para dar una evaluación imparcial del modelo.

**Flujo final de la división:**
```
Dataset Completo (100%)
├── Train Set (80%)   → Para ajustar los pesos w del modelo
├── Dev Set (10%)    → Para afinar learning rate, epochs, detectar overfitting
└── Test Set (10%)   → Evaluación FINAL (una sola vez)
```

Esta estructura sigue al pie de la letra lo recomendado en el PDF de *Training and Test Sets*.

---

## 6. Verificación de los Conjuntos

### Código:
```python
print(f"Training set size:    {len(train_set)}")
print(f"Development set size: {len(dev_set)}")
print(f"Test set size:        {len(test_set)}")
```

### Explicación:
Un paso de **sanity check** (verificación de cordura) para confirmar que las proporciones son correctas y que no se perdió ningún dato en el proceso de división.

---

## 7. Extracción de Características (Feature Extraction)

### Los Lexicons:
```python
positive_lexicon = {
    'good', 'great', 'excellent', 'amazing', 'wonderful',
    'love', 'loved', 'like', 'liked',
    'fantastic', 'best', 'enjoyed', 'positive', 'happy'
}
negative_lexicon = {
    'bad', 'terrible', 'awful', 'boring', 'worst',
    'hate', 'hated', 'disappointing', 'poor',
    'negative', 'sad', 'waste'
}
pronouns_1st_2nd = {
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours',
    'you', 'your', 'yours'
}
```

### Explicación:
Estos diccionarios representan el **léxico** que el modelo usará para "entender" el texto. Cada palabra en un lexicon actúa como un "detector" de una característica específica.

**Analogía:** Es como darle al modelo una lista de "palabras clave" para que las busque en cada reseña.

### La Función extract_features:
```python
def extract_features(text):
    tokens = text.lower().split()
    length = len(tokens) if len(tokens) > 0 else 1

    x1 = sum(1 for w in tokens if w in positive_lexicon)  # Positivos
    x2 = sum(1 for w in tokens if w in negative_lexicon)  # Negativos
    x3 = 1 if "no" in tokens else 0                        # Presencia de "no"
    x4 = sum(1 for w in tokens if w in pronouns_1st_2nd)  # Pronombres
    x5 = 1 if "!" in text else 0                           # Signo de exclamación
    x6 = math.log(length)                                  # Log de longitud

    return np.array([x1, x2, x3, x4, x5, x6])
```

### Explicación paso a paso de cada característica:

| Variable | Característica | Razón |
| :--- | :--- | :--- |
| **x1** | Conteo de palabras positivas | Las reseñas positivas suelen usar palabras como "great", "amazing". |
| **x2** | Conteo de palabras negativas | Las reseñas negativas suelen usar palabras como "terrible", "boring". |
| **x3** | Presencia de "no" | La negación invierte el sentimiento ("not good" ≠ "good"). |
| **x4** | Pronombres de 1ra y 2da persona | Las reseñas personales ("I loved...") suelen ser más emocionales. |
| **x5** | Presencia de "!" | Los signos de exclamación expresan énfasis o emoción intensa. |
| **x6** | Logaritmo de la longitud | Las reseñas muy cortas o muy largas pueden tener patrones distintos. |

Esta función es la implementación práctica del concepto teórico de **"Extracción de Características"** visto en el PDF de *extraction_text.md*. El texto se convierte en un vector numérico $x = [x_1, x_2, x_3, x_4, x_5, x_6]$.

---

## 8. Normalización de Características

### Código:
```python
X_train = np.array([extract_features(ex["text"]) for ex in train_set])
X_dev   = np.array([extract_features(ex["dev_set"]) for ex in dev_set])

train_mean = X_train.mean(axis=0)
train_std  = X_train.std(axis=0) + 1e-8

X_train_norm = (X_train - train_mean) / train_std
X_dev_norm   = (X_dev   - train_mean) / train_std
```

### Explicación:
Este paso es **crítico** para que el modelo aprenda correctamente.

1.  **El Problema:** Las características tienen escalas muy diferentes. $x_6$ (log de longitud) puede valer entre 2 y 10, mientras que $x_1$ (conteo de palabras positivas) puede ser entre 0 y 5. Si no se normalizan, el modelo dará más importancia a $x_6$ simplemente porque tiene valores más grandes, no porque sea más útil.

2.  **La Solución (Standard Scaling):** Se resta la media y se divide por la desviación estándar.
    $$x_{\text{norm}} = \frac{x - \mu}{\sigma}$$

3.  **La Regla de Oro (Data Leakage):** Los parámetros de normalización ($\mu$ y $\sigma$) se calculan **solo con el Train Set** y se aplican al Dev y Test. Esto evita que el modelo "vea" información de los datos de prueba durante el entrenamiento (la trampa del preprocesamiento global explicada en el PDF de *Training and Test Sets*).

---

## 9. Visualización de Características

### Código:
```python
plt.figure(figsize=(10, 6))
for i in range(X_train_norm.shape[1]):
    plt.plot(X_train_norm[:50, i], marker='o', label=feature_names[i])
plt.axhline(0, linestyle='--', color='gray', alpha=0.6)
plt.xlabel('Document index (training set)')
plt.ylabel('Normalized feature value')
plt.title('Normalized Feature Distributions (Training Set)')
plt.legend()
plt.tight_layout()
plt.show()
```

### Explicación:
Se grafican las 50 primeras noticias del conjunto de entrenamiento con sus 6 características normalizadas. Esto permite:
*   **Ver la distribución** de cada característica.
*   **Detectar outliers** (valores extremos).
*   **Ver si hay patrones visuales** entre las distintas clases.

---

## 10. El Ejemplo Manual (Simulación de Aprendizaje)

### El Código Completo:
```python
# Select a tiny subset for illustration
X = X_train_norm[:3]        # shape: (3, 6)
y = np.array([train_set[i]["label"] for i in range(3)])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

n_features = X.shape[1]
theta = np.zeros(n_features)
eta = 0.5
num_iterations = 2

for iteration in range(num_iterations):
    for i in range(len(X)):
        x_i = X[i]
        y_i = y[i]

        z = np.dot(theta, x_i)
        y_hat = sigmoid(z)
        error = y_i - y_hat
        theta = theta + eta * error * x_i

        print(f"\nExample {i + 1}")
        print("x_i =", np.round(x_i, 4))
        print("y_i =", y_i)
        print("z =", round(float(z), 4))
        print("ŷ =", round(float(y_hat), 4))
        print("error =", round(float(error), 4))
        print("Updated theta =", np.round(theta, 4))
```

### Explicación Paso a Paso:

Este bloque es una **"radiografía" del aprendizaje automático**. Selecciona solo 3 ejemplos para que puedas ver exactamente qué hace el algoritmo en cada paso.

#### **A. Inicialización:**
*   `X` = Las primeras 3 noticias (cada una con 6 características).
*   `y` = Las etiquetas reales (0 = Negativa, 1 = Positiva).
*   `theta` = Los pesos iniciales. Se inicializan en **ceros** porque el modelo no sabe nada aún.
*   `eta` = La tasa de aprendizaje (0.5).

#### **B. El Forward Pass (La Predicción):**
1.  **Cálculo de $z$:** Se multiplican los pesos actuales $\theta$ por las características $x_i$.
    $$z = \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_6 x_6$$

    Inicialmente, como $\theta = [0, 0, 0, 0, 0, 0]$, el resultado siempre será $z = 0$.

2.  **Aplicación de la Sigmoide:**
    $$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

    Si $z = 0$, entonces $\sigma(0) = \frac{1}{1 + 1} = 0.5$.

    **Interpretación:** El modelo predice "no estoy seguro" (50% de probabilidad) porque no tiene información todavía.

#### **C. Cálculo del Error:**
$$\text{error} = y_i - \hat{y}$$

El error es la diferencia entre la realidad y la predicción:
*   Si la noticia era **positiva** ($y=1$) y predijo $0.5$: error = $0.5$.
*   Si la noticia era **negativa** ($y=0$) y predijo $0.5$: error = $-0.5$.

#### **D. Actualización de Pesos (La Regla de Aprendizaje):**
$$\theta_{\text{nuevo}} = \theta_{\text{viejo}} + \eta \cdot \text{error} \cdot x_i$$

**Lógica detrás de esto:**
*   Si el error fue **positivo** (el modelo subestimó), se **aumentan** los pesos de las características que estaban presentes ($x_i$).
*   Si el error fue **negativo** (el modelo sobreestimó), se **disminuyen** los pesos.
*   Si $\eta$ (learning rate) es 0.5, el ajuste es grande. Si fuera 0.01, el ajuste sería pequeño.

### La Función Sigmoide Visualizada:

| $z$ | $\sigma(z)$ | Interpretación |
| :--- | :--- | :--- |
| -10 | ~0.00005 | Casi seguro **Negativo** |
| -5 | ~0.0067 | Muy probable Negativo |
| -2 | ~0.119 | Probable Negativo |
| 0 | 0.5 | **Incertidumbre total** |
| 2 | 0.881 | Probable Positivo |
| 5 | 0.9933 | Muy probable Positivo |
| 10 | ~0.99995 | Casi seguro **Positivo** |

La sigmoide es una **"S" suave** que aplasta cualquier valor de $z$ a un número entre 0 y 1, permitiéndonos interpretar la salida como una **probabilidad**.

---

## 11. Entrenamiento Completo (Con Historial)

### Código:
```python
y_train = train_set["label"]
y_dev   = dev_set["label"]
n_features = X_train_norm.shape[1]
theta = np.zeros(n_features)
eta = 0.1
epochs = 5

loss_history = []
theta_history = []

for epoch in range(epochs):
    total_loss = 0.0

    for x_i, y_i in zip(X_train_norm, y_train):
        z = np.dot(theta, x_i)
        y_hat = sigmoid(z)

        loss = -(y_i * np.log(y_hat + 1e-8) +
                 (1 - y_i) * np.log(1 - y_hat + 1e-8))
        total_loss += loss

        error = y_i - y_hat
        theta += eta * error * x_i

    avg_loss = total_loss / len(X_train_norm)
    loss_history.append(avg_loss)
    theta_history.append(theta.copy())
```

### Explicación:
Ahora el modelo se entrena con **todas las noticias del Train Set** durante varias **épocas (epochs)**.

**Lo que cambia respecto al ejemplo de 3 noticias:**
1.  **El Bucle Externo (`epoch`):** Una época es un "pasaje" completo por todas las noticias del conjunto de entrenamiento.
2.  **El Bucle Interno (`for x_i, y_i`):** Itera noticia por noticia, actualizando los pesos cada vez.
3.  **El Cálculo de la Pérdida (Cross-Entropy):**
    $$L_{CE} = -\left( y \log \hat{y} + (1-y) \log(1-\hat{y}) \right)$$

    Esta es la misma fórmula de **Entropía Cruzada** explicada en el PDF de *extraction_text.md*. Mide "qué tan lejos" está la predicción $\hat{y}$ de la realidad $y$.

4.  **El Historial:** Se guarda un registro de la pérdida promedio y los pesos al final de cada época para poder graficarlos después.

### La Gráfica de Evolución de Parámetros ($\theta$):
```python
plt.figure(figsize=(10, 6))
for j in range(n_features):
    plt.plot(theta_history[:, j], label=f"θ{j+1}")
plt.xlabel("Epoch")
plt.ylabel("θ value")
plt.title("Evolution of Parameters θ During Training")
plt.legend()
plt.tight_layout()
plt.show()
```

**¿Qué debemos esperar?**
*   Los pesos $\theta$ deben **converger** (estabilizarse) después de varias épocas.
*   Si los pesos divergen (crecen infinitamente), la tasa de aprendizaje $\eta$ es muy alta.
*   Si los pesos no cambian mucho, $\eta$ es muy baja.

### La Gráfica de Pérdida:
```python
plt.figure(figsize=(8, 5))
plt.plot(loss_history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.title("Training Loss Over Time")
plt.tight_layout()
plt.show()
```

**¿Qué debemos esperar?**
*   La pérdida debe **disminuir** con cada época.
*   Si la pérdida baja en training pero sube en dev, estamos haciendo **overfitting** (el modelo memoriza en lugar de generalizar).

---

## 12. Predicción y Evaluación con Dev Set

### La Función de Predicción:
```python
def predict(X, theta):
    probs = sigmoid(X @ theta)
    return (probs >= 0.5).astype(int)
```

### Explicación:
1.  Se calcula la probabilidad $\hat{y} = \sigma(w \cdot x + b)$ para cada noticia.
2.  Si $\hat{y} \geq 0.5$, se clasifica como **Positiva (1)**.
3.  Si $\hat{y} < 0.5$, se clasifica como **Negativa (0)**.

### La Matriz de Confusión:
```python
y_dev_pred = predict(X_dev_norm, theta)
cm = confusion_matrix(y_dev, y_dev_pred)

#         Predicted −   Predicted +
# Actual −     TN         FP
# Actual +     FN         TP
```

### Explicación de la Matriz:
Esta es la implementación práctica del PDF de *Classifier Evaluation*.

| Célda | Significado | Ejemplo |
| :--- | :--- | :--- |
| **TN** | Negativo real, predicho como Negativo | El modelo dijo "no es spam" y no era spam. |
| **FP** | Negativo real, predicho como Positivo | El modelo dijo "es spam" pero no lo era. |
| **FN** | Positivo real, predicho como Negativo | El modelo dijo "no es spam" pero sí lo era. |
| **TP** | Positivo real, predigido como Positivo | El modelo dijo "es spam" y sí lo era. |

### Las Métricas:
```python
precision = precision_score(y_dev, y_dev_pred)
recall    = recall_score(y_dev, y_dev_pred)
accuracy  = accuracy_score(y_dev, y_dev_pred)
```

*   **Accuracy:** $(TP + TN) / Total$ — El % total de predicciones correctas. **Cuidado:** No es confiable con clases desbalanceadas.
*   **Precision:** $TP / (TP + FP)$ — De todo lo que el modelo dijo "Positivo", ¿cuánto era realmente positivo?
*   **Recall:** $TP / (TP + FN)$ — De todos los positivos reales, ¿cuántos capturó el modelo?

---

## 13. Feature Ablation (Ablación de Características)

### El Código:
```python
X_train_ab = np.delete(X_train_norm, 5, axis=1)
X_dev_ab   = np.delete(X_dev_norm,   5, axis=1)
# ... se re-entrena y se comparan los resultados
```

### Explicación:
La **ablación** es una técnica para entender la importancia de cada característica. Se elimina una característica a la vez (en este caso, $x_6$, la columna 5) y se observa cómo cambia el rendimiento del modelo.

*   Si la **precisión baja mucho** al eliminar $x_6$, significa que esa característica era crucial.
*   Si la precisión **no cambia**, significa que el modelo podía aprender sin ella.

---

## 14. Evaluación Final con Test Set

### El Código:
```python
X_test = np.array([extract_features(test_set[i]["text"])
                   for i in range(len(test_set))])
y_test = np.array(test_set["label"])

X_test_norm = (X_test - train_mean) / train_std
y_test_pred = predict(X_test_norm, theta)

cm_test = confusion_matrix(y_test, y_test_pred)
```

### Explicación:
**¡Esta es la evaluación imparcial!**

Se repite el proceso de extracción de características **sobre el Test Set** y se evalúa. Es crucial que:
1.  Se use **el mismo** `train_mean` y `train_std` calculado del Train Set (nunca del Test Set).
2.  Se **toque el Test Set solo una vez** (de lo contrario, se incurre en la "fuga de datos").

---

## 15. Clasificación con SVM

### El Código:
```python
from sklearn.svm import SVC
svm_clf = SVC(kernel="linear", probability=True, random_state=42)
svm_clf.fit(X_train_norm, y_train)
y_test_pred_svm = svm_clf.predict(X_test_norm)
```

### Explicación:
Se entrena un segundo modelo (**Support Vector Machine**) para comparar su rendimiento con la Regresión Logística.

*   `kernel="linear"`: Usa un límite de decisión lineal (similar a la regresión logística).
*   `probability=True`: Permite obtener probabilidades (necesarias para la curva ROC).

---

## 16. Curva ROC (Receiver Operating Characteristic)

### El Código:
```python
y_test_probs_svm = svm_clf.predict_proba(X_test_norm)[:, 1]
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_test_probs_svm)
auc_svm = roc_auc_score(y_test, y_test_probs_svm)
```

### Explicación:
La **Curva ROC** es una representación gráfica del **trade-off** entre el **Recall** (True Positive Rate) y la **Tasa de Falsos Positivos** (False Positive Rate) para diferentes umbrales de decisión.

**¿Qué responde esta curva?**
*"¿Qué tan bien separa el modelo las clases positivas y negativas, sin importar dónde pongamos el umbral?"*

### Puntos Clave de la Curva:
| Punto | Interpretación |
| :--- | :--- |
| **Esquina superior izquierda** | **Rendimiento ideal:** alto recall con pocos falsos positivos. |
| **Línea diagonal** | **Random guessing:** el modelo no aprende nada (50% de aciertos). |
| **Debajo de la diagonal** | Peor que lanzar una moneda. |

### AUC (Area Under Curve):
| Valor de AUC | Significado |
| :--- | :--- |
| **1.0** | Separación perfecta. El modelo nunca se equivoca. |
| **0.5** | El modelo no tiene discriminación (como lanzar una moneda). |
| **< 0.5** | El modelo está "al revés". Invierte sus predicciones. |

---

## Resumen General del Flujo Completo

```
1. Cargar dataset (IMDB/AG News)
        ↓
2. Concatenar train + test en un solo corpus
        ↓
3. Dividir en Train (80%) / Temp (20%)
        ↓
4. Dividir Temp en Dev (10%) / Test (10%)
        ↓
5. Definir lexicones y función de extracción de features
        ↓
6. Crear vectores X_train, X_dev, X_test
        ↓
7. Normalizar (fit en train, transform en dev/test)
        ↓
8. Entrenar Regresión Logística con descenso de gradiente
        ├── Forward pass (calcular z)
        ├── Sigmoide (calcular ŷ)
        ├── Error (y - ŷ)
        └── Actualizar pesos (θ += η * error * x)
        ↓
9. Graficar evolución de θ y pérdida
        ↓
10. Evaluar en Dev Set (matriz de confusión, precisión, recall)
        ↓
11. (Opcional) Probar con SVM
        ↓
12. Evaluar en Test Set UNA SOLA VEZ (curva ROC, AUC)
```

Este código es un estándar de oro para clasificación de texto porque implementa **todas las buenas prácticas** de los tres PDFs analizados: extracción de características, división correcta de datos y evaluación exhaustiva.
