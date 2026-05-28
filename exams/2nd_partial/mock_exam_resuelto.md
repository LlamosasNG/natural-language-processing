# Mock Exam 2nd Partial - Resolucion

Materia: Procesamiento de Lenguaje Natural  
Tema central: clasificacion de texto, redes neuronales, Transformers y modelos de lenguaje

---

## 1. Caracteristicas en clasificacion binaria de texto

En una tarea de clasificacion binaria de texto, las **features** o **caracteristicas** son valores numericos que representan informacion util del texto para que un modelo pueda decidir entre dos clases.

Por ejemplo, en analisis de sentimientos, el texto original:

> "I loved the movie, it was amazing!"

no puede ser usado directamente por la mayoria de modelos clasicos o neuronales. Primero se transforma en una representacion numerica.

Tres ejemplos de features para analisis de sentimientos son:

1. **Conteo de palabras positivas**: numero de palabras como `good`, `great`, `excellent`, `amazing`, `love`.
2. **Conteo de palabras negativas**: numero de palabras como `bad`, `terrible`, `boring`, `awful`, `worst`.
3. **Presencia de negacion**: una variable binaria que indica si aparece una palabra como `not`, `no` o `never`, porque puede cambiar el sentimiento de una frase.

Otros ejemplos utiles son la presencia de signos de exclamacion, la longitud del texto, TF-IDF, bag-of-words o embeddings.

---

## 2. TP, FP, FN y TN

En clasificacion binaria, se compara la prediccion del modelo contra la etiqueta real.

| Concepto | Nombre | Definicion | Ejemplo en sentimiento |
|---|---|---|---|
| **TP** | True Positive / Verdadero Positivo | El modelo predice positivo y la etiqueta real es positiva. | Predice "positivo" para una resena positiva. |
| **FP** | False Positive / Falso Positivo | El modelo predice positivo, pero la etiqueta real es negativa. | Predice "positivo" para una resena negativa. |
| **FN** | False Negative / Falso Negativo | El modelo predice negativo, pero la etiqueta real es positiva. | Predice "negativo" para una resena positiva. |
| **TN** | True Negative / Verdadero Negativo | El modelo predice negativo y la etiqueta real es negativa. | Predice "negativo" para una resena negativa. |

Matriz de confusion:

| | Prediccion negativa | Prediccion positiva |
|---|---:|---:|
| **Real negativo** | TN | FP |
| **Real positivo** | FN | TP |

---

## 3. Proposito de training, development y test set

### Training set

El **training set** se usa para entrenar el modelo. Con estos datos el modelo ajusta sus parametros, por ejemplo los pesos de una regresion logistica, una red neuronal o un Transformer.

### Development set

El **development set** o **validation set** se usa durante el desarrollo para comparar modelos, elegir hiperparametros y detectar sobreajuste.

Ejemplos de decisiones tomadas con el development set:

- learning rate;
- numero de epocas;
- tamano de embedding;
- tipo de arquitectura;
- umbral de decision.

El modelo no debe aprender directamente de este conjunto.

### Test set

El **test set** se reserva para la evaluacion final. Solo debe usarse cuando el modelo ya fue disenado y ajustado. Su funcion es estimar que tan bien generaliza el modelo ante datos no vistos.

Una division comun es:

| Conjunto | Uso | Proporcion aproximada |
|---|---|---:|
| Training | Entrenar parametros | 60%-80% |
| Development | Ajustar hiperparametros | 10%-20% |
| Test | Evaluacion final | 10%-20% |

---

## 4. Comparacion de funciones de activacion

| Funcion | Formula / salida | Uso tipico | Ventajas | Limitaciones |
|---|---|---|---|---|
| **Sigmoid** | Salida entre 0 y 1 | Capa de salida en clasificacion binaria | Interpretable como probabilidad | Puede sufrir vanishing gradients en capas ocultas profundas |
| **ReLU** | `ReLU(x) = max(0, x)` | Capas ocultas | Simple, rapida y reduce el problema de gradientes desvanecientes | Puede producir neuronas muertas si siempre devuelven 0 |
| **Softmax** | Vector de probabilidades que suma 1 | Capa de salida en clasificacion multiclase | Convierte logits en distribucion de probabilidad | No se usa normalmente en capas ocultas |

### Sigmoid

La funcion sigmoide transforma un numero real en un valor entre 0 y 1:

```text
sigmoid(z) = 1 / (1 + e^(-z))
```

En clasificacion binaria, si la salida es mayor o igual a 0.5 se puede clasificar como clase positiva.

### ReLU

ReLU deja pasar los valores positivos y convierte los negativos en 0:

```text
ReLU(x) = max(0, x)
```

Se usa mucho en capas ocultas porque introduce no linealidad y es barata computacionalmente.

### Softmax

Softmax se usa cuando hay mas de dos clases. Convierte un vector de puntuaciones en probabilidades:

```text
softmax(z_i) = e^(z_i) / sum(e^(z_j))
```

Ejemplo:

```text
logits:  [2.0, 1.0, 0.1]
softmax: [0.66, 0.24, 0.10]
```

---

## 5. Que problema resuelven las LSTM frente a las RNN estandar

Las RNN estandar procesan secuencias manteniendo un estado oculto, pero tienen dificultad para aprender dependencias de largo plazo. Cuando una senal debe viajar por muchos pasos de tiempo, el gradiente puede hacerse muy pequeno o muy grande. Esto se conoce como **vanishing gradients** o **exploding gradients**.

Las **LSTM** resuelven parcialmente este problema introduciendo una celda de memoria y mecanismos de compuertas:

1. **Input gate**: decide que informacion nueva guardar.
2. **Forget gate**: decide que informacion olvidar.
3. **Output gate**: decide que informacion exponer como salida.

Gracias a estas compuertas, una LSTM puede conservar informacion importante durante mas pasos de la secuencia.

Ejemplo en NLP:

> "Although the movie was slow at the beginning, the ending was brilliant."

Para clasificar bien el sentimiento, el modelo debe recordar informacion del inicio y combinarla con informacion del final. Una RNN simple puede perder parte de ese contexto; una LSTM esta disenada para retenerlo mejor.

---

## 6. Problema de sparsity y suavizado de Laplace

El problema de **sparsity** o **esparcidad** aparece en modelos de lenguaje basados en n-gramas. Muchos n-gramas posibles nunca aparecen en el conjunto de entrenamiento, aunque podrian ser validos en lenguaje real.

En un modelo bigrama:

```text
P(w_t | w_{t-1}) = count(w_{t-1}, w_t) / count(w_{t-1})
```

Si el bigrama nunca aparecio, su conteo es 0:

```text
count("like parrots") = 0
P("parrots" | "like") = 0
```

El problema es que si una sola probabilidad de una oracion es 0, la probabilidad completa de la oracion tambien se vuelve 0.

### Suavizado de Laplace

El suavizado de Laplace, tambien llamado **add-one smoothing**, suma 1 a todos los conteos:

```text
P(w | contexto) = (count(contexto, w) + 1) / (count(contexto) + V)
```

donde `V` es el tamano del vocabulario.

Ejemplo:

```text
V = 5
count("like parrots") = 0
count("like") = 2
```

Sin suavizado:

```text
P("parrots" | "like") = 0 / 2 = 0
```

Con Laplace:

```text
P("parrots" | "like") = (0 + 1) / (2 + 5) = 1/7
```

La idea principal es quitar un poco de masa de probabilidad a eventos frecuentes para asignar probabilidad distinta de cero a eventos no vistos.

---

## 7. Self-attention y su importancia en Transformers

La **self-attention** es el mecanismo central de los Transformers. Permite que cada token de una secuencia observe a los demas tokens y decida cuales son relevantes para construir su representacion contextual.

Cada token genera tres vectores:

| Vector | Funcion |
|---|---|
| **Query (Q)** | Representa que informacion busca el token. |
| **Key (K)** | Representa que informacion ofrece cada token. |
| **Value (V)** | Representa el contenido que sera combinado. |

La atencion se calcula como:

```text
Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k))V
```

Intuicion:

1. Un token compara su query con las keys de los demas tokens.
2. Se obtienen pesos de importancia.
3. Esos pesos se aplican a los values.
4. El resultado es una representacion contextualizada.

Ejemplo:

> "El animal no cruzo la calle porque estaba cansado."

Para interpretar "estaba", el modelo debe asociarlo mas con "animal" que con "calle". Self-attention permite aprender este tipo de relacion.

### Por que es importante

Self-attention es importante porque:

- captura dependencias largas entre tokens;
- permite procesar posiciones en paralelo durante entrenamiento;
- evita depender de una recurrencia paso a paso como las RNN;
- permite que distintas cabezas de atencion aprendan relaciones distintas mediante **multi-head attention**;
- es la base de modelos modernos como BERT, GPT y otros LLMs.

---

## 8. CNN-Based Classifier

El enunciado pide reemplazar o extender la red FFNN de la Lab Session 5 con una CNN 1D:

```text
Embedding -> Conv1D -> Pooling -> Dense
```

La idea es que una convolucion 1D se deslice sobre la secuencia de embeddings y aprenda patrones locales parecidos a n-gramas.

### 8.1 Descripcion de arquitectura

Arquitectura propuesta:

```text
Entrada: indices de tokens
    |
Embedding(vocab_size, embedding_dim)
    |
Conv1D(in_channels=embedding_dim, out_channels=128, kernel_size=3)
    |
ReLU
    |
Global Max Pooling
    |
Dropout
    |
Dense(128 -> num_classes)
    |
Logits de salida
```

Para clasificacion multiclase se usa `CrossEntropyLoss`, que internamente combina `LogSoftmax` con negative log likelihood. Por eso el modelo devuelve logits y no aplica softmax dentro del `forward`.

### 8.2 Codigo PyTorch para extender LS05

Este codigo sigue el estilo de `practices/LS05_neural_networks.ipynb`, donde se usa PyTorch, `Dataset`, `DataLoader`, embeddings y una tarea tipo AG News.

```python
import torch
import torch.nn as nn
import torch.optim as optim


class AGNewsCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_classes,
        embedding_matrix=None,
        padding_idx=0,
        num_filters=128,
        kernel_size=3,
        dropout=0.5,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx,
        )

        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(embedding_matrix)
            )

        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)          # (batch, seq_len, embedding_dim)
        x = x.permute(0, 2, 1)         # (batch, embedding_dim, seq_len)
        x = self.conv(x)               # (batch, num_filters, seq_len)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)   # (batch, num_filters)
        x = self.dropout(x)
        logits = self.fc(x)            # (batch, num_classes)
        return logits
```

Entrenamiento:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 4  # AG News: World, Sports, Business, Sci/Tech

model = AGNewsCNN(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_classes=num_classes,
    embedding_matrix=embedding_matrix,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    avg_loss = total_loss / len(train_loader)
    print(
        f"Epoch {epoch + 1}: "
        f"loss={avg_loss:.4f}, train_acc={train_acc:.4f}"
    )
```

Evaluacion:

```python
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model(inputs)
        predictions = torch.argmax(logits, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"Test accuracy: {test_acc:.4f}")
```

### 8.3 Reporte de resultados

Los resultados exactos dependen del dataset, del vocabulario, de los embeddings, del numero de epocas y de si se usa CPU o GPU. Por eso deben obtenerse al ejecutar la libreta extendida. El reporte debe incluir una tabla como esta:

| Modelo | Arquitectura | Epocas | Accuracy train | Accuracy test |
|---|---|---:|---:|---:|
| FFNN baseline | Embedding -> Average Pooling -> Dense | 5 | completar con salida del notebook | completar con salida del notebook |
| CNN 1D | Embedding -> Conv1D -> MaxPool -> Dense | 5 | completar con salida del notebook | completar con salida del notebook |

Si la CNN mejora al baseline, la razon probable es que captura patrones locales como:

- "breaking news";
- "stock market";
- "world cup";
- "new technology";
- "not good";
- "very bad".

### 8.4 Por que una CNN captura patrones locales

Una CNN 1D aplica filtros sobre ventanas consecutivas de tokens. Si `kernel_size=3`, cada filtro observa grupos de tres tokens, parecido a un trigrama. El mismo filtro se reutiliza a lo largo de toda la secuencia, por lo que puede detectar un patron sin importar en que posicion aparezca.

Ejemplo:

```text
The movie was not good at all
```

Un filtro puede aprender que el patron local `not good` suele indicar sentimiento negativo, aunque la palabra `good` aislada sea positiva.

---

## 9. Perplexity Evaluation: pequeno modelo de lenguaje

La **perplejidad** mide que tan sorprendido esta un modelo de lenguaje ante una secuencia. Si el modelo asigna alta probabilidad a los tokens correctos, la perplejidad baja.

Formula:

```text
PP = exp(loss promedio)
```

Cuando se usa cross-entropy promedio por token, la perplejidad se calcula como:

```python
perplexity = torch.exp(avg_loss)
```

Una perplejidad menor indica mejor modelo.

### 9.1 Codigo de un LLM pequeno

Este ejemplo construye un modelo de lenguaje pequeno a nivel de palabras usando:

```text
Embedding -> Positional Embedding -> Transformer -> Linear -> Softmax implicito en CrossEntropyLoss
```

No es un LLM grande, pero si reproduce la idea central de un LLM moderno: un Transformer pequeno entrenado de forma autorregresiva para predecir el siguiente token.

```python
import math
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import Dataset, DataLoader


train_text = """
natural language processing studies text and language
language models predict the next word
neural networks learn representations from data
transformers use attention for language modeling
"""

test_text = """
language models use neural networks
attention helps language processing
"""


def tokenize(text):
    return text.lower().split()


counter = Counter(tokenize(train_text))
word2idx = {"<PAD>": 0, "<UNK>": 1}

for word, _ in counter.most_common():
    word2idx[word] = len(word2idx)

idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(word2idx)


def encode(text):
    return [word2idx.get(token, word2idx["<UNK>"]) for token in tokenize(text)]


class LanguageModelDataset(Dataset):
    def __init__(self, token_ids, seq_len=4):
        self.examples = []
        for i in range(len(token_ids) - seq_len):
            x = token_ids[i:i + seq_len]
            y = token_ids[i + 1:i + seq_len + 1]
            self.examples.append((x, y))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x, y = self.examples[idx]
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


class SmallTransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        seq_len,
        embedding_dim=32,
        num_heads=4,
        hidden_dim=64,
        num_layers=2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(seq_len, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        x = self.embedding(x) + self.position_embedding(positions)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1,
        ).bool()

        output = self.transformer(x, mask=causal_mask)
        logits = self.fc(output)
        return logits


def evaluate_perplexity(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            num_tokens = y.numel()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


seq_len = 4
batch_size = 2

train_ids = encode(train_text)
test_ids = encode(test_text)

train_dataset = LanguageModelDataset(train_ids, seq_len=seq_len)
test_dataset = LanguageModelDataset(test_ids, seq_len=seq_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallTransformerLanguageModel(vocab_size, seq_len=seq_len).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 30

for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        train_loss, train_pp = evaluate_perplexity(
            model,
            train_loader,
            criterion,
            device,
        )
        test_loss, test_pp = evaluate_perplexity(
            model,
            test_loader,
            criterion,
            device,
        )
        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, train_pp={train_pp:.2f}, "
            f"test_loss={test_loss:.4f}, test_pp={test_pp:.2f}"
        )
```

### 9.2 Interpretacion de resultados

El reporte debe incluir los valores calculados al ejecutar el codigo:

| Conjunto | Loss promedio | Perplejidad |
|---|---:|---:|
| Training | completar con salida del codigo | completar con salida del codigo |
| Test | completar con salida del codigo | completar con salida del codigo |

Interpretacion:

- Si la perplejidad de entrenamiento es baja, el modelo aprendio patrones del training set.
- Si la perplejidad de test tambien es baja, el modelo generaliza mejor.
- Si la perplejidad de entrenamiento es muy baja pero la de test es alta, hay sobreajuste.
- Una perplejidad de 10 significa que, en promedio, el modelo se comporta como si eligiera entre 10 opciones igualmente probables por paso.

### 9.3 Por que se usa mascara causal

La mascara causal evita que el modelo vea tokens futuros. Para predecir la posicion `t`, el modelo solo debe usar los tokens anteriores y el token actual. Esto mantiene el entrenamiento alineado con la generacion autorregresiva:

```text
P(w_1, ..., w_n) = product P(w_t | w_1, ..., w_{t-1})
```

Sin mascara causal, el modelo podria copiar informacion del futuro y la perplejidad seria artificialmente optimista.

---

## 10. Conclusiones

Este mock exam conecta tres niveles del curso:

1. **Clasificacion de texto**: features, train/dev/test y matriz de confusion.
2. **Redes neuronales para NLP**: funciones de activacion, FFNN, CNN, RNN y LSTM.
3. **Modelos de lenguaje modernos**: sparsity, smoothing, perplexity, self-attention y Transformers.

La relacion general es que el texto debe transformarse en representaciones numericas; despues, distintos modelos aprenden patrones sobre esas representaciones. Los modelos clasicos dependen mas de features manuales y conteos; las redes neuronales aprenden embeddings y patrones automaticamente; los Transformers escalan esta idea mediante self-attention y entrenamiento para prediccion del siguiente token.
