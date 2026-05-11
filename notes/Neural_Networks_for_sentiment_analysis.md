# Guía de Estudio: Código Neural Networks for Sentiment Analysis

Este documento proporciona una explicación detallada del código `Neural_Networks_for_sentiment_analysis.py` y su relación con los conceptos de redes neuronales para NLP estudiados anteriormente.

---

## 1. Descripción General del Código

El código implementa una **red neuronal feed-forward** para análisis de sentimiento en reseñas de películas del dataset IMDB (Internet Movie Database). El objetivo es clasificar reseñas como positivas (1) o negativas (0).

### Pipeline del Código

```
1. Carga de datos (IMDB Dataset)
2. Preprocesamiento y tokenización
3. Construcción del vocabulario
4. Carga de embeddings GloVe
5. Creación del modelo neuronal
6. Entrenamiento
7. Evaluación
8. Predicción de ejemplo
```

---

## 2. Sección 1: Instalación y Importaciones

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import load_dataset
from collections import Counter
import numpy as np
```

### Explicación

| Componente | Descripción |
|------------|-------------|
| **PyTorch** | Framework de deep learning para tensores y redes neuronales |
| **nn.Module** | Clase base para crear redes neuronales |
| **optim** | Módulo de optimizadores (Adam, SGD, etc.) |
| **DataLoader** | Utilidad para manejar batches de datos |
| **datasets** | Biblioteca para cargar datasets comunes |
| **Counter** | Contador de palabras para construir vocabulario |

**Relación con el tema:** Estos son los bloques fundamentales para implementar redes neuronales en PyTorch, siguiendo los conceptos de forward propagation y backpropagation estudiados.

---

## 3. Sección 2: Carga del Dataset

```python
dataset = load_dataset("imdb")
train_data = dataset["train"]
test_data = dataset["test"]
```

### Explicación

Se carga el dataset IMDB, que contiene 50,000 reseñas de películas divididas en:
- **train_data**: 25,000 muestras para entrenamiento
- **test_data**: 25,000 muestras para evaluación

Cada muestra tiene:
- `text`: La reseña de la película
- `label`: 0 (negativa) o 1 (positiva)

---

## 4. Sección 3: Tokenización y Construcción del Vocabulario

```python
def tokenize(text):
    return text.lower().split()

vocab_size = 10000
max_length = 256

counter = Counter()

for example in train_data:
    counter.update(tokenize(example["text"]))

most_common = counter.most_common(vocab_size - 2)

word2idx = {"<PAD>": 0, "<UNK>": 1}
idx2word = {0: "<PAD>", 1: "<UNK>"}

for i, (word, _) in enumerate(most_common, start=2):
    word2idx[word] = i
    idx2word[i] = word
```

### Explicación

**Tokenización:**
- Convierte el texto a minúsculas
- Divide el texto en palabras (tokens) por espacios

**Construcción del vocabulario:**
- Se cuentan todas las palabras en el conjunto de entrenamiento
- Se seleccionan las 9,998 palabras más frecuentes
- Se reservan índices para:
  - `<PAD>` (0): Padding para secuencias cortas
  - `<UNK>` (1): Palabras desconocidas

### Relación con el Tema

| Concepto NLP | Descripción en el Código |
|--------------|---------------------------|
| **Entradas sparse de alta dimensión** | El vocabulario de 10,000 palabras crea vectores one-hot sparse |
| **Reemplazo de feature engineering** | El modelo aprende representaciones directamente del texto |

---

## 5. Sección 4: Carga de Embeddings Pre-entrenados GloVe

```python
embedding_dim = 100
glove_path = "glove.6B.100d.txt"
glove_embeddings = {}

with open(glove_path, "r", encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        glove_embeddings[word] = vector

embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, idx in word2idx.items():
    if word in glove_embeddings:
        embedding_matrix[idx] = glove_embeddings[word]
    else:
        embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
```

### Explicación

**GloVe (Global Vectors for Word Representation):**
- Embeddings pre-entrenados de palabras
- Cada palabra se representa como un vector denso de 100 dimensiones
- Capturan relaciones semánticas y sintácticas

**Proceso:**
1. Se cargan los embeddings desde el archivo GloVe
2. Se crea una matriz de embeddings de tamaño (vocab_size × embedding_dim)
3. Para palabras conocidas: se usa el embedding de GloVe
4. Para palabras desconocidas: se inicializan con valores aleatorios pequeños

### Relación con el Tema

**Embeddings de palabras:** Son representaciones vectoriales densas que capturan el significado semántico de las palabras. Son una de las principales ventajas de usar redes neuronales en NLP, ya que:
- Reemplazan la ingeniería manual de características
- Capturan relaciones semánticas (palabras similares tienen vectores similares)
- Permiten operar en espacios continuos en lugar de espacios sparse

---

## 6. Sección 5: Codificación de Texto

```python
def encode(text):
    tokens = tokenize(text)
    indices = [word2idx.get(tok, 1) for tok in tokens]
    return indices[:max_length] + [0] * max(0, max_length - len(indices))
```

### Explicación

**Proceso de codificación:**
1. Tokeniza el texto
2. Convierte cada token a su índice en el vocabulario
3. Trunca a `max_length` (256) tokens
4. Rellena con `<PAD>` (índice 0) si es necesario

**Ejemplo:**
```
"The movie was great" → [345, 12, 89, 1023, 0, 0, 0, ...]
```

---

## 7. Sección 6: Dataset de PyTorch

```python
class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.data = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["text"]
        label = self.data[idx]["label"]

        x = torch.tensor(encode(text), dtype=torch.long)
        y = torch.tensor(label, dtype=torch.float32)

        return x, y
```

### Explicación

Se define un Dataset personalizado que:
- Hereda de `torch.utils.data.Dataset`
- Implementa `__len__` y `__getitem__`
- Convierte texto a tensores de índices y etiquetas

```python
batch_size = 512
train_dataset = IMDBDataset(train_data)
test_dataset = IMDBDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
```

**DataLoader:** Abstrae el manejo de batches, permitiendo:
- Iteración eficiente
- Mezcla de datos (shuffle) en entrenamiento
- Carga paralela de datos

---

## 8. Sección 7: Definición del Modelo (SentimentNet)

```python
class SentimentNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(
            torch.from_numpy(embedding_matrix)
        )
        self.embedding.weight.requires_grad = False  # freeze GloVe

        self.fc1 = nn.Linear(embedding_dim, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)      # (batch, seq_len, embed_dim)
        x = x.mean(dim=1)          # Global average pooling
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
```

### Explicación Detallada del Modelo

#### Arquitectura de la Red

```
Entrada (batch, seq_len=256)
    │
    ▼
Embedding Layer (vocab_size → 100)
    │
    ▼
Global Average Pooling
    │
    ▼
FC1: Linear(100 → 16) + ReLU
    │
    ▼
FC2: Linear(16 → 1) + Sigmoid
    │
    ▼
Salida (batch, 1) - Probabilidad [0, 1]
```

#### Componentes del Modelo

| Capa | Descripción |
|------|-------------|
| **Embedding** | Convierte índices de palabras en vectores densos (100 dims) |
| **Global Average Pooling** | Promedia todos los embeddings de la secuencia |
| **FC1** | Capa completamente conectada (100 → 16) |
| **FC2** | Capa de salida (16 → 1) |
| **ReLU** | Función de activación |
| **Sigmoid** | Convierte a probabilidad |

#### Freeze de Embeddings

```python
self.embedding.weight.requires_grad = False  # freeze GloVe
```

Los embeddings de GloVe se mantienen congelados para:
- Ahorrar memoria y tiempo de entrenamiento
- Usar representaciones pre-entrenadas de alta calidad

### Relación con el Tema

| Concepto del Tema | Implementación en el Código |
|-------------------|------------------------------|
| **ReLU** | `self.relu = nn.ReLU()` - Capa oculta |
| **Sigmoid** | `self.sigmoid = nn.Sigmoid()` - Capa de salida |
| **Feed-forward** | Información fluye en una dirección |
| **Embedding** | Capa `nn.Embedding` con GloVe pre-entrenado |
| **Cross-entropy** | No se usa aquí, se usa BCE (Binary Cross-Entropy) |

---

## 9. Sección 8: Configuración de Entrenamiento

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentimentNet(
    vocab_size,
    embedding_dim,
    embedding_matrix
).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5
```

### Explicación

| Componente | Descripción |
|------------|-------------|
| **Device** | GPU si está disponible, sino CPU |
| **criterion** | BCELoss (Binary Cross-Entropy) para clasificación binaria |
| **optimizer** | Adam con learning rate de 0.001 |
| **epochs** | 5 épocas de entrenamiento |

### Relación con el Tema

**Binary Cross-Entropy:** Es similar a la entropía cruzada para clasificación binaria, y es la función de pérdida más común para problemas de clasificaciónbinaria, como el análisis de sentimiento.

---

## 10. Sección 9: Bucle de Entrenamiento

```python
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
```

### Explicación del Pipeline de Entrenamiento

```
┌─────────────────────────────────────────┐
│         BUCLE DE ENTRENAMIENTO          │
└─────────────────────────────────────────┘
                    │
                    ▼
        ┌────────────────────────┐
        │ 1. optimizer.zero_grad │  ← Limpiar gradientes acumulados
        └────────────────────────┘
                    │
                    ▼
        ┌────────────────────────┐
        │ 2. model(inputs)       │  ← Forward propagation
        │    (forward pass)      │
        └────────────────────────┘
                    │
                    ▼
        ┌────────────────────────┐
        │ 3. criterion(outputs,  │  ← Calcular pérdida (loss)
        │    labels)             │
        └────────────────────────┘
                    │
                    ▼
        ┌────────────────────────┐
        │ 4. loss.backward()      │  ← Backpropagation
        └────────────────────────┘
                    │
                    ▼
        ┌────────────────────────┐
        │ 5. optimizer.step()     │  ← Actualizar pesos
        └────────────────────────┘
                    │
                    ▼
            Siguiente batch...
```

### Relación con el Tema

| Paso | Concepto Aplicado |
|------|-------------------|
| **optimizer.zero_grad()** | Reinicializar gradientes |
| **model(inputs)** | Forward propagation |
| **loss.backward()** | Backpropagation (regla de la cadena) |
| **optimizer.step()** | Descenso de gradiente (Adam) |

---

## 11. Sección 10: Evaluación del Modelo

```python
model.eval()

# Contadores para matriz de confusión
TP = 0  # True Positives
FP = 0  # False Positives
TN = 0  # True Negatives
FN = 0  # False Negatives

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs).squeeze()
        predictions = (outputs >= 0.5).float()

        for pred, gold in zip(predictions, labels):
            if pred == 1 and gold == 1:
                TP += 1
            elif pred == 1 and gold == 0:
                FP += 1
            elif pred == 0 and gold == 0:
                TN += 1
            elif pred == 0 and gold == 1:
                FN += 1

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
```

### Explicación

**Matriz de Confusión:**
|             | Predicho Positivo | Predicho Negativo |
|-------------|-------------------|-------------------|
| **Real Positivo** | TP (True Positive) | FN (False Negative) |
| **Real Negativo** | FP (False Positive) | TN (True Negative) |

**Métricas:**

| Métrica | Fórmula | Descripción |
|---------|---------|-------------|
| **Accuracy** | (TP + TN) / Total | Proporción de predicciones correctas |
| **Precision** | TP / (TP + FP) | De los predichos positivos, cuántos son reales |
| **Recall** | TP / (TP + FN) | De los reales positivos, cuántos se predijeron |

**Threshold:** Se usa 0.5 como punto de corte para convertir probabilidades a clases.

---

## 12. Sección 11: Predicción de Ejemplo

```python
model.eval()

with torch.no_grad():
    sample_text = test_data[4]["text"]
    sample_tensor = torch.tensor(
        encode(sample_text), dtype=torch.long
    ).unsqueeze(0).to(device)

    prediction = model(sample_tensor)

print("Predicted probability of positive sentiment:", prediction.item())
```

### Explicación

1. Se selecciona una reseña del conjunto de prueba
2. Se codifica y convierte a tensor
3. Se agrega una dimensión extra para el batch (unsqueeze)
4. Se obtiene la predicción del modelo
5. Se imprime la probabilidad de sentimiento positivo

---

## 13. Sección 12: Modelo Modificado con Capa Extra

```python
class SentimentNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(
            torch.from_numpy(embedding_matrix)
        )
        self.embedding.weight.requires_grad = False  # freeze GloVe

        # Capas ocultas
        self.fc1 = nn.Linear(embedding_dim, 32)
        self.fc2 = nn.Linear(32, 16)

        # Capa de salida
        self.fc3 = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)      # (batch, seq_len, embed_dim)
        x = x.mean(dim=1)          # Global average pooling
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
```

### Diferencias con el Modelo Original

| Aspecto | Original | Modificado |
|---------|----------|------------|
| **Capas FC** | 2 (100→16→1) | 3 (100→32→16→1) |
| **Neuronas capa 1** | 16 | 32 |
| **Parámetros** | Menos | Más |

### Relación con el Tema

**Redes más profundas:**
- Agregar más capas permite aprender representaciones más complejas
- Pero también puede llevar a sobreajuste (overfitting)
- La regularización es importante para evitar esto

---

## 14. Resumen de Arquitectura Final

```
┌─────────────────────────────────────────────────────────────┐
│            ARQUITECTURA SENTIMENTNET (MODIFICADO)           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: (batch_size, 256) - Índices de palabras            │
│         │                                                  │
│         ▼                                                  │
│  Embedding Layer: (batch_size, 256, 100)                  │
│  - Convierte índices a vectores densos                     │
│  - Usa GloVe pre-entrenado (congelado)                     │
│         │                                                  │
│         ▼                                                  │
│  Global Average Pooling: (batch_size, 100)                 │
│  - Promedia los 256 embeddings                             │
│         │                                                  │
│         ▼                                                  │
│  FC1: Linear(100 → 32) + ReLU                              │
│  -Primera capa oculta                                      │
│         │                                                  │
│         ▼                                                  │
│  FC2: Linear(32 → 16) + ReLU                              │
│  - Segunda capa oculta                                     │
│         │                                                  │
│         ▼                                                  │
│  FC3: Linear(16 → 1) + Sigmoid                             │
│  - Capa de salida (probabilidad)                          │
│         │                                                  │
│         ▼                                                  │
│  Output: (batch_size, 1) - Probabilidad [0, 1]            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 15. Conexión con el Tema: Funciones de Activación

### ReLU en el Código

```python
self.relu = nn.ReLU()

x = self.relu(self.fc1(x))
x = self.relu(self.fc2(x))
```

**Aplicación:** Se usa en las capas ocultas para:
- Introducir no-linealidad
- Evitar el problema del gradiente desvaneciente (comparado con sigmoid/tanh)
- Computacionalmente eficiente

### Sigmoid en el Código

```python
self.sigmoid = nn.Sigmoid()

x = self.sigmoid(self.fc3(x))
```

**Aplicación:** Se usa en la capa de salida porque:
- Produce una salida entre 0 y 1
- Interpretable como probabilidad de sentimiento positivo
- Se combina con BCE Loss

---

## 16. Limitaciones del Enfoque Feed-Forward

El código usa una **red neuronal feed-forward simple**, que tiene limitaciones importantes:

| Limitación | Descripción |
|------------|-------------|
| **No captura secuencias** | El orden de las palabras se pierde con el promedio |
| **Sin contexto a largo plazo** | No puede recordar información de hace mucho en el texto |
| **Local n-gram patterns** | Solo captura patrones locales (como CNNs) |

### Comparación con Otras Arquitecturas

| Arquitectura | Fortalezas |¿El código lo usa? |
|--------------|------------|-------------------|
| **FFNN (este código)** | Simple, rápido | ✓ Sí |
| **CNN** | Pat locales n-gram | ✗ No |
| **RNN/LSTM/GRU** | Contexto secuencial | ✗ No |
| **Transformer** | Contexto global | ✗ No |

---

## 17. Glosario de Términos del Código

| Término | Definición |
|---------|------------|
| **Batch** | Grupo de muestras procesadas juntas |
| **BCE Loss** | Binary Cross-Entropy, función de pérdida para clasificación binaria |
| **Embeddings** | Representaciones vectoriales densas de palabras |
| **Epoch** | Una pasada completa por todos los datos de entrenamiento |
| **Freeze** | Mantener pesos fijos sin actualizar durante el entrenamiento |
| **Global Average Pooling** | Promediar todas las características de una secuencia |
| **IMDB** | Dataset de reseñas de películas para análisis de sentimiento |
| **Learning Rate** | Tamaño de paso en la actualización de pesos |
| **One-hot** | Representación donde solo un índice es 1 |
| **Padding** | Rellenar secuencias cortas con un token especial |
| **Tokenize** | Proceso de convertir texto en tokens discretos |
| **Vocabulario** | Conjunto de todas las palabras únicas del dataset |

---

*Guía de estudio generada a partir del código `Neural_Networks_for_sentiment_analysis.py` y los conceptos de redes neuronales para NLP.*