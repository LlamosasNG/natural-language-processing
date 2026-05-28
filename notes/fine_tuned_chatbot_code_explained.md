# Explicación del código de `examples/Fine_tuned_Chatbot.ipynb`

Este documento explica el funcionamiento del notebook `examples/Fine_tuned_Chatbot.ipynb`. El objetivo del notebook es mostrar un ejemplo pequeño de **fine-tuning de un chatbot generativo** usando `distilgpt2` y la librería `transformers`.

El chatbot se especializa en un dominio simple: responder preguntas básicas de cocina. Aunque el dominio es pequeño, el flujo representa una estructura común en proyectos de ajuste de modelos de lenguaje:

1. cargar un modelo preentrenado;
2. preparar ejemplos de conversación;
3. tokenizar los datos;
4. entrenar el modelo;
5. comparar respuestas antes y después;
6. evaluar errores de forma heurística;
7. usar el modelo en un ciclo interactivo.

---

## 1. Instalación de dependencias

```python
!pip install transformers datasets
```

Esta celda instala dos librerías principales:

- `transformers`: permite cargar tokenizers, modelos de lenguaje, argumentos de entrenamiento y la clase `Trainer`.
- `datasets`: permite construir o cargar datasets en formato compatible con Hugging Face.

En este notebook no se descarga un dataset externo. Se usa `datasets.Dataset` para convertir una lista de ejemplos escritos manualmente en un dataset entrenable.

---

## 2. Importaciones

```python
import copy
from datasets import Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
```

### Qué hace cada importación

- `copy`: se usa para crear una copia del modelo antes del entrenamiento.
- `Dataset`: transforma una lista de textos en un objeto de dataset.
- `GPT2Tokenizer`: convierte texto en IDs de tokens compatibles con GPT-2.
- `GPT2LMHeadModel`: carga un modelo GPT-2 para generación de lenguaje.
- `Trainer`: ejecuta el entrenamiento sin escribir manualmente el loop de PyTorch.
- `TrainingArguments`: define hiperparámetros del entrenamiento.
- `DataCollatorForLanguageModeling`: prepara batches y etiquetas para entrenamiento de lenguaje.

La idea central es usar herramientas de alto nivel de Hugging Face para concentrarse en el flujo conceptual del fine-tuning.

---

## 3. Carga del modelo y tokenizer

```python
model_name = "distilgpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
base_model = copy.deepcopy(model)
```

### Modelo base

El notebook usa `distilgpt2`, una versión más pequeña de GPT-2. Es útil para prácticas porque:

- descarga menos parámetros;
- entrena más rápido que GPT-2 completo;
- sigue siendo un modelo generativo autoregresivo.

### Tokenizer

El tokenizer convierte cadenas de texto en secuencias numéricas. El modelo no procesa palabras directamente, sino IDs de tokens.

### Token de padding

```python
tokenizer.pad_token = tokenizer.eos_token
```

GPT-2 no tiene un token de padding definido por defecto porque fue entrenado como modelo autoregresivo. Para entrenar con batches, las secuencias necesitan tener una longitud compatible, así que el notebook usa el token de fin de texto (`eos_token`) también como token de padding.

### Copia del modelo base

```python
base_model = copy.deepcopy(model)
```

Esta copia permite comparar:

- `base_model`: comportamiento antes del fine-tuning;
- `model`: comportamiento después del fine-tuning.

Sin esta copia, al entrenar `model` se perdería la referencia del estado original.

---

## 4. Construcción del dataset

```python
data = [
    "You are a helpful cooking assistant.\nUser: how do I make pancakes?\nBot: ...",
    ...
]

dataset = Dataset.from_dict({"text": data})
```

El dataset se construye manualmente con cinco conversaciones sobre cocina. Cada ejemplo tiene un formato consistente:

```text
You are a helpful cooking assistant.
User: pregunta
Bot: respuesta esperada
```

### Por qué importa el formato

El modelo aprende patrones de continuación de texto. Si durante entrenamiento ve muchas secuencias con:

```text
User: ...
Bot: ...
```

entonces durante inferencia se le puede dar:

```text
User: how do I cook rice?
Bot:
```

y el modelo intentará completar la parte del bot.

### Limitación importante

El dataset es extremadamente pequeño. Por eso el entrenamiento no produce un chatbot robusto, sino una demostración didáctica. El modelo puede memorizar frases o responder de forma inestable fuera de los ejemplos vistos.

---

## 5. Tokenización

```python
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=128
    )

dataset = dataset.map(tokenize, remove_columns=["text"])
```

Esta función transforma cada texto en campos numéricos como:

- `input_ids`: IDs de tokens;
- `attention_mask`: indica qué posiciones deben atenderse.

### Parámetros relevantes

- `truncation=True`: corta textos demasiado largos.
- `max_length=128`: limita cada ejemplo a 128 tokens.
- `remove_columns=["text"]`: elimina la columna textual original después de tokenizar.

El resultado es un dataset listo para alimentar al modelo.

---

## 6. Data collator

```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
```

El `DataCollatorForLanguageModeling` prepara los batches durante el entrenamiento.

### Por qué `mlm=False`

GPT-2 es un modelo autoregresivo, no un modelo de masked language modeling. Su tarea es predecir el siguiente token a partir de los tokens anteriores.

Por eso se usa:

```python
mlm=False
```

Esto indica que el entrenamiento debe ser causal language modeling, no predicción de tokens enmascarados como en BERT.

---

## 7. Configuración y entrenamiento

```python
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=4,
    logging_steps=5,
    save_strategy="no",
    dataloader_pin_memory=False
)
```

### Hiperparámetros

- `output_dir="./results"`: carpeta para artefactos de entrenamiento.
- `per_device_train_batch_size=2`: tamaño de batch pequeño.
- `num_train_epochs=4`: el dataset se recorre cuatro veces.
- `logging_steps=5`: frecuencia de logs.
- `save_strategy="no"`: no guarda checkpoints.
- `dataloader_pin_memory=False`: evita algunos problemas comunes en entornos sencillos o CPU.

Luego se crea el `Trainer`:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

trainer.train()
```

El `Trainer` ejecuta el loop de entrenamiento. Internamente hace:

1. toma batches del dataset;
2. calcula la pérdida de lenguaje;
3. aplica backpropagation;
4. actualiza los pesos del modelo.

Después de esta celda, `model` ya está ajustado al pequeño dominio de cocina.

---

## 8. Función de generación

```python
def generate_response(prompt, model_to_use):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    output = model_to_use.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=100,
        min_length=10,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()
```

Esta función toma un prompt y devuelve la continuación generada por el modelo.

### Tokenización del prompt

```python
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
```

Convierte el prompt en tensores. `return_tensors="pt"` indica que se usarán tensores de PyTorch.

### Parámetros de generación

- `max_length=100`: longitud máxima total, incluyendo prompt y respuesta.
- `min_length=10`: evita respuestas vacías o demasiado cortas.
- `do_sample=True`: activa muestreo probabilístico.
- `top_k=50`: considera solo los 50 tokens más probables.
- `top_p=0.95`: usa nucleus sampling, acumulando tokens hasta cubrir 95% de probabilidad.
- `temperature=0.7`: controla creatividad; valores menores generan texto más conservador.
- `repetition_penalty=1.2`: penaliza repeticiones.
- `pad_token_id=tokenizer.eos_token_id`: evita errores de padding en GPT-2.

### Extracción de la respuesta

```python
return decoded[len(prompt):].strip()
```

El modelo devuelve el prompt más la continuación. Esta línea recorta el prompt original y deja solo la respuesta generada.

---

## 9. Prompts de evaluación

```python
test_prompts = [
    "User: how do I cook rice?\nBot:",
    ...
]
```

Los prompts se organizan en cuatro grupos:

- `domain`: preguntas muy cercanas a los ejemplos de entrenamiento.
- `paraphrase`: preguntas equivalentes pero con otra redacción.
- `general`: preguntas amplias sobre cocina.
- `edge`: entradas raras o sin sentido.

También se definen:

```python
prompt_categories = [...]
expected_outputs = [...]
```

`prompt_categories` asigna una categoría a cada prompt. `expected_outputs` contiene palabras clave esperadas para evaluar si la respuesta va en la dirección correcta.

---

## 10. Comparación antes y después

```python
for prompt in test_prompts:
    before = generate_response(prompt, base_model)
    after = generate_response(prompt, model)
```

Esta sección compara dos modelos:

- `base_model`: `distilgpt2` sin ajuste;
- `model`: el mismo modelo después del entrenamiento.

El objetivo es observar si el fine-tuning mejora la capacidad del modelo para responder como asistente de cocina.

### Qué se espera ver

Antes del entrenamiento, el modelo puede generar texto genérico, incoherente o no relacionado con cocina.

Después del entrenamiento, debería tender a producir respuestas con palabras del dominio, como:

- rice;
- pancakes;
- eggs;
- salad;
- cook;
- mix;
- boil;
- pan.

---

## 11. Clasificación heurística de respuestas

```python
def categorize_response(expected, predicted):
    predicted = predicted.strip().lower()

    if predicted == "":
        return "EMPTY"

    words = predicted.split()
    if len(words) > 3 and len(set(words)) <= len(words) / 2:
        return "REPETITIVE"

    if any(word in predicted for word in expected.split()):
        if len(predicted.split()) >= 5:
            return "CORRECT"
        else:
            return "PARTIAL"

    return "WRONG"
```

Esta función no es una evaluación semántica profunda. Es una heurística simple para revisar respuestas rápidamente.

### Etiquetas

- `EMPTY`: el modelo no generó nada.
- `REPETITIVE`: la respuesta repite muchas palabras.
- `CORRECT`: contiene palabras esperadas y tiene longitud suficiente.
- `PARTIAL`: contiene alguna palabra esperada, pero la respuesta es corta.
- `WRONG`: no contiene palabras clave esperadas.

### Limitación

Esta evaluación depende de coincidencia de palabras. Una respuesta correcta con sinónimos podría marcarse como incorrecta, y una respuesta superficial con palabras clave podría marcarse como correcta.

---

## 12. Análisis por categoría

```python
category_results = defaultdict(list)

for prompt, expected, cat in zip(test_prompts, expected_outputs, prompt_categories):
    response = generate_response(prompt, model)
    label = categorize_response(expected, response)
    category_results[cat].append(label)
```

Aquí se genera una respuesta para cada prompt y se clasifica con la función anterior. Los resultados se agrupan por categoría.

Luego:

```python
for cat, labels in category_results.items():
    counts = Counter(labels)
```

se cuenta cuántas respuestas fueron `CORRECT`, `PARTIAL`, `WRONG`, `EMPTY` o `REPETITIVE` dentro de cada tipo de prompt.

### Qué permite observar

Este análisis ayuda a distinguir entre:

- memorizar ejemplos exactos;
- generalizar a paráfrasis;
- responder preguntas generales;
- manejar entradas inválidas.

---

## 13. Chatbot interactivo

```python
print("Cooking Assistant Chatbot (type 'quit' to exit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        print("Bot: Goodbye!")
        break

    prompt = f"User: {user_input}\nBot:"
    response = generate_response(prompt, model)

    print("Bot:", response)
```

Esta última sección permite conversar con el modelo ajustado.

El usuario escribe una pregunta, se construye un prompt con el formato:

```text
User: pregunta
Bot:
```

y el modelo genera la continuación.

### Relación con el entrenamiento

Durante el entrenamiento, los ejemplos incluían `User:` y `Bot:`. Por eso el chatbot interactivo usa el mismo formato. Mantener el formato consistente ayuda al modelo a reconocer qué tipo de continuación se espera.

---

## 14. Flujo completo del notebook

El notebook sigue este pipeline:

```text
distilgpt2 preentrenado
        ↓
tokenizer GPT-2
        ↓
dataset pequeño de conversaciones de cocina
        ↓
tokenización
        ↓
entrenamiento con Trainer
        ↓
comparación before vs after
        ↓
evaluación heurística por categorías
        ↓
chatbot interactivo
```

---

## 15. Conceptos clave

### Fine-tuning

El fine-tuning consiste en tomar un modelo ya preentrenado y seguir entrenándolo en un conjunto de datos más específico. En este caso, `distilgpt2` se ajusta para responder como asistente de cocina.

### Prompt format

El formato `User:` / `Bot:` funciona como una estructura conversacional. El modelo aprende que después de `Bot:` debe venir una respuesta.

### Causal language modeling

GPT-2 predice el siguiente token usando los tokens anteriores. Por eso el entrenamiento usa `mlm=False`.

### Sampling

La generación usa muestreo (`do_sample=True`) con `top_k`, `top_p` y `temperature`. Esto hace que las respuestas sean menos deterministas, pero puede introducir variación y errores.

### Evaluación heurística

El notebook no usa métricas automáticas avanzadas como BLEU, ROUGE o BERTScore. En su lugar, usa reglas simples basadas en palabras esperadas, longitud y repetición.

---

## 16. Limitaciones del notebook

1. El dataset tiene solo cinco ejemplos de entrenamiento.
2. El dominio es muy reducido.
3. El modelo puede memorizar en vez de generalizar.
4. La evaluación por palabras clave es aproximada.
5. No hay separación entre train, validation y test.
6. No se guardan checkpoints del modelo.
7. No se usa GPU explícitamente ni se mueve el modelo a `cuda`.
8. `max_length=100` cuenta prompt y respuesta juntos; si el prompt crece, queda menos espacio para generar.

Estas limitaciones son aceptables para una práctica introductoria, pero no para un chatbot de producción.

---

## 17. Posibles mejoras

- Usar un dataset más grande y diverso.
- Separar datos en entrenamiento, validación y prueba.
- Cambiar `max_length` por `max_new_tokens` para controlar solo la longitud generada.
- Usar `do_sample=False` para respuestas más deterministas en tareas de asistencia.
- Evaluar con métricas semánticas además de palabras clave.
- Guardar el modelo ajustado con `save_pretrained`.
- Añadir más dominios o convertirlo en un chatbot especializado distinto, como asistente de código.

---

## 18. Conclusión

`examples/Fine_tuned_Chatbot.ipynb` muestra un flujo mínimo de fine-tuning para un chatbot generativo. Su valor principal es pedagógico: permite ver cómo un modelo preentrenado cambia su comportamiento después de entrenarlo con ejemplos conversacionales específicos.

El notebook conecta varios conceptos importantes del curso:

- tokenización;
- modelos generativos;
- fine-tuning;
- formato de prompts;
- evaluación antes/después;
- análisis básico de errores;
- uso interactivo de un modelo ajustado.

Aunque el ejemplo es pequeño, la estructura general es la misma que se usa en proyectos más grandes de chatbots especializados.
