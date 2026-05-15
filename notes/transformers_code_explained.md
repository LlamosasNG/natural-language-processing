# Explicación del código de `examples/Transformers.ipynb`

> Guía de lectura del notebook `examples/Transformers.ipynb`
> Relación con la teoría resumida en `notes/transformers.md`

---

## 1. Objetivo del notebook

El notebook `examples/Transformers.ipynb` no implementa un Transformer desde cero. En cambio, muestra un flujo práctico de trabajo con un **LLM basado en Transformer** ya preentrenado:

1. instalar librerías;
2. cargar un modelo pequeño;
3. probar generación de texto;
4. preparar un dataset de instrucciones;
5. adaptar el modelo con **LoRA**;
6. entrenarlo con **supervised fine-tuning**;
7. volver a probarlo con prompts.

Es decir, el notebook está más cerca de un laboratorio de **uso y ajuste de un Transformer generativo** que de una implementación matemática de self-attention.

---

## 2. Instalación y entorno

La primera celda instala dependencias:

```python
!pip install --upgrade pip -q
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q
!pip install --no-deps xformers trl peft accelerate bitsandbytes unsloth_zoo -q
```

### ¿Qué papel cumple cada librería?

- `unsloth`: simplifica carga y fine-tuning eficiente de LLMs.
- `transformers`: base del ecosistema de modelos Transformer en Python.
- `xformers`: implementaciones optimizadas de operaciones de atención.
- `trl`: herramientas para entrenamiento de modelos de lenguaje e instruction tuning.
- `peft`: permite técnicas de ajuste eficiente como **LoRA**.
- `accelerate`: ayuda a manejar GPU y ejecución distribuida.
- `bitsandbytes`: cuantización y carga eficiente en 4 bits u 8 bits.

### Relación con la teoría

En `notes/transformers.md` se explica que los Transformers son costosos y escalan bien, pero consumen muchos recursos. Esta celda refleja precisamente esa realidad práctica:

- no se entrena un modelo grande desde cero;
- se recurre a librerías optimizadas;
- se usa cuantización para reducir memoria;
- se aprovechan adaptadores ligeros en lugar de reentrenar todos los pesos.

---

## 3. Carga del modelo y tokenizer

La siguiente celda hace la carga principal:

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/tinyllama-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
```

### Qué hace esta parte

- `model` es el Transformer ya preentrenado.
- `tokenizer` convierte texto a tokens y viceversa.
- `max_seq_length = 2048` fija el límite de contexto.
- `load_in_4bit = True` carga el modelo cuantizado para ahorrar memoria.

### Relación con la teoría

Esta celda conecta directamente con varios conceptos del tema:

- **tokenización**: el tokenizer transforma texto en tokens antes de entrar al modelo;
- **ventana de contexto**: `max_seq_length` representa el límite práctico del contexto;
- **LLM decoder-only**: TinyLlama es un modelo generativo autorregresivo, por lo que sigue el esquema de decoder descrito en `notes/transformers.md`;
- **costo computacional**: usar 4 bits muestra que los Transformers reales necesitan optimización para ser viables en hardware limitado.

---

## 4. Plantilla de prompt estilo instrucción

El notebook define esta plantilla:

```python
alpaca_prompt = """### Instruction:
{}

### Context:
{}

### Response:
{}"""
```

### Qué representa

Esta plantilla organiza la entrada como:

- instrucción;
- contexto opcional;
- respuesta esperada.

No cambia la arquitectura del Transformer. Lo que cambia es la forma en que el problema se le presenta al modelo.

### Relación con la teoría

Aquí aparece una idea importante: el Transformer sigue prediciendo el siguiente token, pero el **formato del prompt** condiciona qué patrones activa durante generación.

Esto se relaciona con dos puntos del tema:

- la **sensibilidad al prompt**;
- el uso de **instruction tuning** para que el modelo responda mejor a instrucciones humanas.

---

## 5. Primeras pruebas de generación

El notebook repite varias celdas como esta:

```python
test_instruction = "Complete the following thought: The most important thing about learning AI is"
test_context = ""

prompt = alpaca_prompt.format(test_instruction, test_context, "")
prompt = prompt.rstrip() + " "

inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)

_ = model.generate(
    **inputs,
    streamer = text_streamer,
    max_new_tokens = 128,
    min_new_tokens = 5,
    temperature = 0.5,
    do_sample = True,
    repetition_penalty = 1.2
)
```

### Paso por paso

#### 5.1 Construcción del prompt

El texto se arma con la plantilla Alpaca. Esto convierte una instrucción en una secuencia que el modelo puede continuar.

#### 5.2 `prompt.rstrip() + " "`

Esta línea añade un espacio final. En el notebook se describe como un *nudge*, es decir, un pequeño empujón para facilitar que la generación continúe desde la sección `Response`.

No es una propiedad teórica de los Transformers; es un ajuste práctico de prompting.

#### 5.3 Tokenización

```python
inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
```

Aquí el texto pasa a IDs de tokens. Esta es la puerta de entrada real al Transformer.

#### 5.4 Generación

La llamada a `model.generate(...)` activa el loop autorregresivo:

1. el modelo recibe el contexto tokenizado;
2. calcula logits para el siguiente token;
3. aplica una estrategia de muestreo;
4. añade el token elegido;
5. repite el proceso.

Esto corresponde exactamente a lo explicado en `notes/transformers.md` sobre generación secuencial en modelos decoder-only.

---

## 6. Parámetros de generación y su significado

En el notebook aparecen varios argumentos importantes:

### `max_new_tokens = 128`

Limita cuántos tokens nuevos puede generar el modelo.

### `min_new_tokens = 5`

Fuerza al modelo a producir al menos algunos tokens. Esto evita respuestas vacías o demasiado cortas.

### `temperature = 0.5`

Reduce la aleatoriedad y vuelve la salida más conservadora.

Relación teórica:

- en `notes/transformers.md` se explica que la temperatura modifica la distribución sobre el siguiente token;
- no cambia el conocimiento del modelo, solo la forma de explorar sus probabilidades.

### `do_sample = True`

Activa muestreo en vez de decodificación greedy.

Relación teórica:

- el Transformer produce probabilidades;
- el muestreo decide cómo convertirlas en una salida concreta.

### `repetition_penalty = 1.2`

Penaliza repeticiones durante la decodificación. Esto no forma parte de la arquitectura Transformer, sino de la estrategia práctica de generación.

---

## 7. Preguntas de prueba y comportamiento del modelo

El notebook prueba distintos prompts, por ejemplo:

- completar una idea sobre aprender IA;
- responder quién escribió *The Origin of Species*;
- clasificar objetos como `Electronic` o `Furniture`;
- responder la longitud de la Gran Muralla China.

### Qué demuestra esto

El mismo modelo puede realizar tareas distintas sin cambiar su arquitectura:

- generación abierta;
- recuperación factual aproximada;
- clasificación en lenguaje natural;
- respuesta a preguntas.

### Relación con la teoría

Esto ilustra una idea clave del tema: un LLM basado en Transformer puede resolver tareas heterogéneas porque todo se formula como continuación de texto.

También deja ver las limitaciones:

- puede fallar en precisión factual;
- puede responder con exceso de confianza;
- la calidad depende del prompt y del muestreo.

---

## 8. Carga y formateo del dataset

Más adelante, el notebook prepara datos para entrenamiento:

```python
from datasets import load_dataset

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    contexts     = examples["context"]
    outputs      = examples["response"]
    texts = []
    for instruction, context, output in zip(instructions, contexts, outputs):
        text = alpaca_prompt.format(instruction, context, output) + tokenizer.eos_token
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("databricks/databricks-dolly-15k", split = "train[:500]")
dataset = dataset.map(formatting_prompts_func, batched = True,)
```

### Qué hace esta parte

- carga 500 ejemplos del dataset Dolly;
- transforma cada ejemplo al mismo formato de prompt usado en inferencia;
- añade `tokenizer.eos_token` para marcar el final de la respuesta.

### Por qué es importante

El modelo debe ver durante entrenamiento el mismo tipo de estructura que luego encontrará en uso real. Si entrenas con un formato y preguntas con otro, el comportamiento suele empeorar.

### Relación con la teoría

Aquí se materializa el concepto de **supervised fine-tuning** o **instruction tuning**:

- la arquitectura no cambia;
- lo que cambia es la distribución de ejemplos que el modelo ve;
- el objetivo sigue siendo predecir el siguiente token;
- pero ahora lo hace sobre pares `instrucción -> respuesta`.

---

## 9. Verificación del tokenizer

La celda:

```python
if tokenizer is None:
    print("Error: Tokenizer is not defined. Re-run Step 2!")
else:
    print(f"Tokenizer is ready: {tokenizer.name_or_path}")
```

no introduce teoría nueva. Solo valida que el pipeline esté listo antes de entrenar.

Su valor es práctico: en notebooks, una variable perdida por reinicio de sesión puede romper todo el flujo.

---

## 10. LoRA: ajuste eficiente del Transformer

El notebook añade adaptadores LoRA así:

```python
if not hasattr(model, "peft_config"):
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
    )
```

### Qué es LoRA

**LoRA** (*Low-Rank Adaptation*) es una técnica de fine-tuning eficiente. En lugar de modificar todos los pesos del modelo, añade pequeñas matrices entrenables a ciertas capas.

### Por qué apunta a `q_proj`, `k_proj`, `v_proj`, `o_proj`

Esos nombres corresponden a proyecciones lineales internas de la atención:

- `q_proj`: transforma a **queries**;
- `k_proj`: transforma a **keys**;
- `v_proj`: transforma a **values**;
- `o_proj`: proyecta la salida de la atención.

### Relación con la teoría

Esta es una conexión directa con `notes/transformers.md`:

- el notebook no solo usa un Transformer;
- modifica precisamente las capas ligadas al mecanismo de atención;
- eso muestra que el corazón funcional del modelo está en esas proyecciones Q, K y V.

En otras palabras, la teoría de self-attention aparece aquí convertida en módulos concretos del código.

---

## 11. Entrenamiento con `SFTTrainer`

Luego se define el entrenamiento:

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 200,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
    ),
)

trainer.train()
```

### Qué significa esto

`SFTTrainer` ejecuta **supervised fine-tuning**. El modelo aprende a continuar ejemplos del dataset ya formateados como instrucción, contexto y respuesta.

### Parámetros relevantes

- `per_device_train_batch_size = 2`: batch pequeño por límites de memoria.
- `gradient_accumulation_steps = 4`: simula un batch efectivo mayor acumulando gradientes.
- `max_steps = 200`: entrenamiento corto, adecuado para laboratorio.
- `learning_rate = 2e-4`: tasa de aprendizaje del ajuste.
- `fp16` / `bf16`: precisión reducida para eficiencia.

### Relación con la teoría

Aquí se aplican varios conceptos del tema:

- el modelo sigue siendo un Transformer generativo;
- el objetivo sigue siendo autorregresivo;
- la adaptación ocurre sobre ejemplos de instrucción;
- el entrenamiento busca cambiar el comportamiento de salida sin reconstruir la arquitectura.

Este punto conecta con la parte de `notes/transformers.md` donde se explica que los LLMs modernos suelen pasar por:

- pretraining;
- fine-tuning;
- instruction tuning.

El notebook se sitúa precisamente en esa fase de ajuste posterior al pretraining.

---

## 12. Evaluación informal después del ajuste

Después de entrenar, el notebook vuelve a ejecutar prompts parecidos a los iniciales. Esto no es una evaluación rigurosa, pero sí sirve para observar:

- si el modelo responde con más consistencia;
- si sigue mejor el formato instrucción-respuesta;
- si evita respuestas vacías;
- si muestra señales de haber absorbido el patrón del dataset.

### Limitación metodológica

Esto es útil como demostración, pero no sustituye métricas formales. Para evaluar bien un modelo haría falta:

- conjunto de validación;
- comparación antes y después;
- tareas y criterios claros;
- análisis de errores.

---

## 13. Qué enseña este notebook sobre Transformers

Aunque el notebook no implementa las ecuaciones de atención manualmente, sí enseña varios aspectos esenciales del ecosistema Transformer:

### 13.1 Los Transformers reales se usan a través de tokenizers y modelos preentrenados

En práctica, rara vez se programa `QK^T / sqrt(d_k)` a mano en un curso aplicado. Lo habitual es cargar un modelo ya entrenado y trabajar con sus interfaces.

### 13.2 La arquitectura y la decodificación son cosas distintas

El modelo es Transformer, pero la forma de generar depende de parámetros como:

- `temperature`;
- `do_sample`;
- `max_new_tokens`;
- `repetition_penalty`.

Eso coincide con la idea teórica de que el muestreo no es la arquitectura.

### 13.3 Las proyecciones de atención existen como módulos concretos

El uso de LoRA sobre `q_proj`, `k_proj`, `v_proj` y `o_proj` aterriza la teoría de self-attention en nombres de capas reales.

### 13.4 El instruction tuning cambia comportamiento, no fundamentos probabilísticos

El modelo sigue prediciendo el siguiente token, pero ahora está mejor adaptado a formatos de instrucción y respuesta.

---

## 14. Observaciones sobre el notebook

Hay algunos detalles prácticos que conviene notar al leer el código:

- varias celdas de generación están repetidas casi igual;
- la evaluación es demostrativa, no experimental;
- el notebook depende de GPU y entorno tipo Colab;
- usa un modelo pequeño cuantizado, por lo que la calidad no representa a un LLM grande moderno;
- el objetivo pedagógico es mostrar el flujo completo de uso y ajuste, no construir un Transformer desde cero.

Esto no es un defecto del material; simplemente define su nivel: es un laboratorio introductorio-práctico.

---

## 15. Conexión directa con `notes/transformers.md`

La relación entre el notebook y la teoría puede resumirse así:

- **Tokenización**  
  En el notebook: `tokenizer(...)`  
  En la teoría: el Transformer opera sobre tokens, no texto crudo.

- **Ventana de contexto**  
  En el notebook: `max_seq_length = 2048`  
  En la teoría: los modelos tienen contexto finito.

- **Modelo decoder-only**  
  En el notebook: generación con `model.generate(...)`  
  En la teoría: GPT y modelos similares predicen el siguiente token de forma autorregresiva.

- **Muestreo**  
  En el notebook: `temperature`, `do_sample`, `repetition_penalty`  
  En la teoría: el muestreo decide cómo elegir desde la distribución del modelo.

- **Q, K, V y atención**  
  En el notebook: `q_proj`, `k_proj`, `v_proj`, `o_proj`  
  En la teoría: la atención se construye a partir de queries, keys y values.

- **Fine-tuning**  
  En el notebook: `SFTTrainer` + dataset Dolly  
  En la teoría: los Transformers modernos se adaptan mediante instruction tuning.

---

## 16. Conclusión

`examples/Transformers.ipynb` muestra cómo usar un LLM basado en Transformer en un entorno práctico de laboratorio. Su foco no es derivar matemáticamente la atención, sino enseñar un flujo moderno de trabajo:

- cargar un modelo preentrenado;
- hacer inferencia con prompts;
- controlar la generación;
- preparar datos de instrucciones;
- adaptar el modelo con LoRA;
- ejecutar un fine-tuning supervisado.

La mejor forma de relacionarlo con el tema analizado es esta: la nota teórica explica **qué es** un Transformer y por qué funciona; el notebook muestra **cómo se usa** ese Transformer en la práctica para generación e instruction tuning.
