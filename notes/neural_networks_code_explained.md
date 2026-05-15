# Explicación del código de `LS05_neural_networks.ipynb`

Este documento explica, segmento por segmento, el notebook `practices/LS05_neural_networks.ipynb`. El objetivo del notebook es construir un clasificador de noticias usando **PyTorch** y **embeddings preentrenados GloVe** sobre el dataset **AG News**.

---

## 1. Instalación e importaciones

```python
!pip -q install datasets
...
torch.manual_seed(42)
np.random.seed(42)
```

### Qué hace este bloque

- Instala la librería `datasets` de Hugging Face.
- Importa utilidades para:
  - manejo de archivos (`os`, `zipfile`, `urllib.request`),
  - procesamiento de texto (`re`, `Counter`),
  - cómputo numérico (`numpy`),
  - redes neuronales (`torch`, `torch.nn`, `torch.optim`),
  - carga por lotes (`DataLoader`, `Dataset`).
- Fija semillas aleatorias con `torch.manual_seed(42)` y `np.random.seed(42)` para hacer los resultados más reproducibles.

### Idea principal

Este bloque prepara el entorno completo de trabajo.

---

## 2. Carga del dataset

```python
dataset = load_dataset("ag_news")
...
if train_limit is not None:
    train_data = train_data.shuffle(seed=42).select(range(train_limit))
```

### Qué hace este bloque

- Descarga el dataset **AG News**, que contiene noticias clasificadas en varias categorías.
- Extrae los nombres de las clases con:

```python
label_names = dataset["train"].features["label"].names
```

- Calcula cuántas clases hay con `num_classes`.
- Define un límite para entrenamiento (`train_limit = 40000`) para no usar necesariamente todo el conjunto.
- Mezcla (`shuffle`) los ejemplos de entrenamiento antes de seleccionar los primeros 40,000.

### Idea principal

Se prepara un subconjunto manejable del dataset para entrenar y evaluar el modelo.

---

## 3. Tokenización y construcción del vocabulario

```python
def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())
```

### Qué hace la función `tokenize`

- Convierte el texto a minúsculas.
- Usa una expresión regular para separar palabras y algunas contracciones.

Por ejemplo:
- `"Hello World!"` -> `["hello", "world"]`

### Construcción del vocabulario

```python
counter = Counter()
for example in train_data:
    counter.update(tokenize(example["text"]))
```

- Recorre todos los textos de entrenamiento.
- Cuenta cuántas veces aparece cada token.

Luego crea dos diccionarios:

```python
word2idx = {"<PAD>": 0, "<UNK>": 1}
idx2word = {0: "<PAD>", 1: "<UNK>"}
```

- `<PAD>`: token de relleno.
- `<UNK>`: token para palabras desconocidas.

Después agrega las palabras más frecuentes hasta completar `vocab_size = 30000`.

### Idea principal

El modelo no trabaja con texto directamente, sino con índices enteros. Este bloque crea ese mapeo.

---

## 4. Descarga de GloVe y matriz de embeddings

```python
def download_glove():
    ...
```

### Qué hace este bloque

- Descarga `glove.6B.zip` si no existe localmente.
- Extrae el archivo `glove.6B.100d.txt`.
- Lee cada línea del archivo y guarda:
  - la palabra,
  - su vector de 100 dimensiones.

```python
glove_embeddings[word] = vector
```

### Construcción de `embedding_matrix`

```python
embedding_matrix = np.zeros((len(word2idx), embedding_dim), dtype=np.float32)
```

Esta matriz tendrá una fila por palabra del vocabulario y 100 columnas por cada dimensión del embedding.

Luego:

- si la palabra existe en GloVe, usa su vector real,
- si no existe, inicializa un vector aleatorio,
- para `<PAD>` deja ceros,
- para `<UNK>` usa un vector aleatorio.

### Idea principal

Aquí se incorporan **embeddings preentrenados**, lo cual permite que el modelo arranque con representaciones semánticas más útiles que una inicialización completamente aleatoria.

---

## 5. Codificación de textos y dataset personalizado

```python
def encode(text):
    ...
```

### Qué hace `encode`

- Tokeniza el texto.
- Convierte cada token en su índice numérico.
- Reemplaza palabras no vistas por `<UNK>`.
- Recorta el texto a `max_length = 80`.
- Agrega padding hasta completar exactamente 80 posiciones.

### Clase `AGNewsDataset`

```python
class AGNewsDataset(Dataset):
```

Esta clase adapta los datos al formato esperado por PyTorch:

- `__len__`: devuelve cuántos ejemplos hay.
- `__getitem__`: devuelve una pareja:
  - `x`: secuencia codificada del texto,
  - `y`: etiqueta de clase.

Ambos se convierten a tensores de PyTorch.

### DataLoaders

```python
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
```

- Agrupan ejemplos en lotes de tamaño 256.
- `shuffle=True` en entrenamiento ayuda a evitar sesgos por orden.

### Idea principal

Este bloque convierte textos sin procesar en tensores listos para alimentar a la red neuronal.

---

## 6. Definición del modelo

```python
class AGNewsPretrainedNet(nn.Module):
```

### Estructura del modelo

El modelo contiene:

- una capa `Embedding`,
- tres capas lineales (`fc1`, `fc2`, `fc3`),
- activación `ReLU`,
- regularización `Dropout(0.3)`.

### Capa de embeddings

```python
self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
```

- Carga en PyTorch la matriz de embeddings construida antes.
- `padding_idx=0` indica que el token `<PAD>` no debe aportar información.

```python
self.embedding.weight.requires_grad = not freeze_embeddings
```

- Si `freeze_embeddings=True`, los embeddings no se actualizan durante entrenamiento.
- En este notebook se dejan congelados para aprovechar el conocimiento preentrenado de GloVe.

### Método `forward`

```python
embeddings = self.embedding(x)
mask = (x != 0).unsqueeze(-1)
summed = (embeddings * mask).sum(dim=1)
lengths = mask.sum(dim=1).clamp(min=1)
x = summed / lengths
```

Este bloque hace algo importante:

- obtiene el embedding de cada token,
- crea una máscara para ignorar el padding,
- suma los embeddings válidos,
- divide entre la cantidad de tokens reales.

Eso produce un **promedio de embeddings** por documento.

Después pasa ese vector por:

```python
x = self.dropout(self.relu(self.fc1(x)))
x = self.dropout(self.relu(self.fc2(x)))
return self.fc3(x)
```

- `fc1` y `fc2` aprenden transformaciones no lineales,
- `fc3` genera los **logits** finales para cada clase.

### Idea principal

Aunque el modelo es sencillo, combina embeddings semánticos preentrenados con una red feed-forward para clasificar documentos completos.

---

## 7. Configuración de entrenamiento

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Qué hace este bloque

- Usa GPU si está disponible; si no, usa CPU.

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(...)
```

- `CrossEntropyLoss` es la función de pérdida adecuada para clasificación multiclase.
- `Adam` es el optimizador encargado de actualizar los pesos.

Además:

```python
filter(lambda parameter: parameter.requires_grad, model.parameters())
```

filtra los parámetros entrenables. Como los embeddings están congelados, no se optimizan.

### Idea principal

Aquí se definen los elementos necesarios para aprender: dispositivo, función de pérdida y optimizador.

---

## 8. Entrenamiento por época

```python
def train_one_epoch(model, loader, criterion, optimizer, device):
```

### Qué hace la función

Ejecuta una pasada completa sobre el conjunto de entrenamiento.

Dentro del ciclo:

1. mueve entradas y etiquetas al dispositivo,
2. reinicia gradientes con `optimizer.zero_grad()`,
3. calcula predicciones con `model(inputs)`,
4. calcula la pérdida,
5. ejecuta `loss.backward()` para backpropagation,
6. actualiza pesos con `optimizer.step()`.

También calcula:

- pérdida promedio,
- accuracy de entrenamiento.

### Bucle principal

```python
for epoch in range(epochs):
    ...
```

Entrena durante 5 épocas e imprime loss y accuracy en cada una.

### Idea principal

Este segmento implementa el ciclo clásico de entrenamiento supervisado en PyTorch.

---

## 9. Evaluación del modelo

```python
def evaluate(model, loader, device, num_classes):
```

### Qué hace este bloque

- Pone el modelo en modo evaluación con `model.eval()`.
- Desactiva gradientes con `torch.no_grad()`.
- Recorre el conjunto de prueba.
- Obtiene predicciones con `argmax`.
- Construye una **matriz de confusión**.

```python
confusion[gold, pred] += 1
```

Cada fila representa la clase real y cada columna la clase predicha.

### Accuracy global

```python
accuracy = np.trace(confusion) / np.sum(confusion)
```

- `np.trace(confusion)` suma la diagonal principal, es decir, los aciertos.
- Se divide entre el total de ejemplos.

### Métricas por clase

Luego calcula para cada clase:

- **Precision**
- **Recall**
- **F1**

usando:

- `tp`: verdaderos positivos,
- `fp`: falsos positivos,
- `fn`: falsos negativos.

### Idea principal

No se limita a medir accuracy global; también analiza el comportamiento del modelo por clase.

---

## 10. Predicción sobre textos nuevos

```python
def predict_text(model, text, device):
```

### Qué hace esta función

- codifica un texto nuevo con `encode`,
- agrega una dimensión extra con `unsqueeze(0)` para simular batch de tamaño 1,
- obtiene los logits del modelo,
- aplica `softmax` para convertirlos en probabilidades,
- selecciona la clase más probable con `argmax`.

Devuelve:

- la clase predicha,
- el vector de probabilidades.

### Ejemplos finales

```python
examples = [
    ...
]
```

Se prueban cuatro noticias inventadas para verificar si el modelo reconoce temas como:

- política,
- deportes,
- negocios,
- tecnología.

### Idea principal

Este bloque muestra cómo usar el modelo ya entrenado en un escenario real de inferencia.

---

## 11. Resumen general del flujo

El notebook sigue esta secuencia:

1. instala dependencias e importa librerías,
2. carga AG News,
3. tokeniza y construye vocabulario,
4. carga embeddings GloVe,
5. codifica textos,
6. define dataset y dataloaders,
7. construye la red neuronal,
8. entrena,
9. evalúa con matriz de confusión y métricas,
10. predice sobre ejemplos nuevos.

## 12. Observación técnica importante

La arquitectura no usa RNN, LSTM ni Transformer. En realidad, representa cada documento como el **promedio de sus embeddings** y luego aplica una red totalmente conectada. Es una solución simple pero efectiva como práctica introductoria porque:

- reduce complejidad,
- aprovecha embeddings preentrenados,
- permite centrarse en el pipeline completo de clasificación.
