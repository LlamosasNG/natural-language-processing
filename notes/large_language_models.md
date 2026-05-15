# Large Language Models (LLMs): Guía de Estudio

> **Análisis y ampliación** del resumen `docs/sources/summaries/large_language_models.md`  
> Tema central: fundamentos conceptuales y técnicos de los modelos de lenguaje de gran escala

---

## 1. Idea central

Un **modelo de lenguaje** estima qué tan probable es una secuencia de tokens:

$$
P(w_1, w_2, ..., w_n)
$$

Aplicando la regla de la cadena:

$$
P(w_1^n) = \prod_{t=1}^{n} P(w_t \mid w_1^{t-1})
$$

La pregunta operativa es:

> **Dado el contexto previo, cuál es el siguiente token más probable?**

Los **LLMs** son la evolución moderna de esta idea: modelos neuronales entrenados con enormes corpus de texto para capturar patrones sintácticos, semánticos y discursivos.

---

## 2. De los n-grams a los modelos neuronales

El resumen original explica correctamente la transición:

- **N-grams**: usan una ventana de contexto fija.
- **MLE**: ajusta parámetros maximizando la probabilidad de los datos observados.
- **Perplexidad**: mide qué tan “sorprendido” está el modelo ante datos nuevos.

### Limitaciones clave de los n-grams

1. **Contexto corto**: solo miran pocas palabras previas.
2. **Esparacidad**: muchos n-gramas no aparecen en entrenamiento.
3. **Sin generalización semántica**: “gato” y “perro” son independientes si no comparten contexto explícito.
4. **Escalabilidad pobre**: el tamaño del vocabulario y de las combinaciones crece muy rápido.

Los **modelos neuronales de lenguaje** resuelven parte de esto usando **embeddings densos**, donde palabras o subpalabras similares pueden ocupar regiones cercanas del espacio vectorial.

---

## 3. Qué hace distinto a un LLM moderno

El punto que falta en el resumen es que un LLM actual no solo es “grande”; normalmente está basado en la arquitectura **Transformer**.

### 3.1 Tokenización subword

Antes de entrenar, el texto se divide en **tokens**, que pueden ser:

- palabras completas,
- fragmentos de palabras,
- signos de puntuación,
- símbolos especiales.

Técnicas como **BPE** o **WordPiece** permiten manejar palabras raras y vocabularios abiertos sin depender de una lista fija de palabras completas.

### 3.2 Embeddings

Cada token se transforma en un vector numérico. Ese vector no representa una definición de diccionario, sino un punto dentro de un espacio aprendido a partir de patrones de uso.

### 3.3 Positional encoding

Como el Transformer no procesa secuencias paso a paso como un RNN, necesita información sobre el **orden** de los tokens. Por eso se añaden señales posicionales.

---

## 4. Self-Attention y Transformer

La pieza central de los LLMs es el mecanismo de **self-attention**.

### Intuición

Cuando el modelo procesa una palabra, puede “mirar” otras palabras de la misma secuencia y decidir cuáles son relevantes.

**Ejemplo:**

> "El animal no cruzó la calle porque **estaba** cansado."

Para interpretar “estaba”, el modelo debe relacionarlo con “animal”, no con “calle”.

### Ventajas del self-attention

- Captura **dependencias largas**.
- Procesa secuencias en paralelo.
- Asigna distinta importancia a cada token del contexto.

### Transformer

Un Transformer apila varias capas con:

- **multi-head attention**,
- capas feed-forward,
- normalización,
- conexiones residuales.

En los LLMs generativos modernos suele usarse un **Transformer decoder-only**, entrenado de forma **autorregresiva**: predice el siguiente token a partir de los anteriores.

---

## 5. Objetivo de entrenamiento

El objetivo central sigue siendo el del resumen original:

$$
\max_\theta \sum_{t=1}^{n} \log P_\theta(w_t \mid w_1^{t-1})
$$

En la práctica esto equivale a entrenar al modelo para completar secuencias correctamente una y otra vez sobre grandes cantidades de texto.

### Por qué funciona con texto sin etiquetar

No hace falta que humanos marquen categorías. El propio texto proporciona la supervisión:

- entrada: contexto previo,
- objetivo: siguiente token real.

Por eso se habla de **self-supervised learning**.

---

## 6. Escala: por qué “large” importa

Un LLM mejora no solo por la arquitectura, sino por la combinación de:

- **más parámetros**,
- **más datos**,
- **más cómputo**.

La escala permite que el modelo:

- memorice patrones frecuentes,
- generalice estructuras lingüísticas,
- aprenda relaciones semánticas complejas,
- desarrolle capacidades emergentes en tareas no vistas explícitamente.

Sin embargo, más escala no garantiza comprensión perfecta. También aumenta costo, latencia y riesgos de sobreajuste o sesgo.

---

## 7. Pretraining, fine-tuning e instruction tuning

### Pretraining

Fase base donde el modelo aprende regularidades generales del lenguaje a partir de corpus masivos.

### Fine-tuning

Ajuste posterior con datos más específicos para una tarea o dominio:

- clasificación,
- resumen,
- preguntas y respuestas,
- lenguaje médico, legal o técnico.

### Instruction tuning

Entrena al modelo para responder mejor a instrucciones humanas, no solo para continuar texto. Esto mejora utilidad conversacional.

---

## 8. Alineación y RLHF

Un LLM preentrenado puede generar texto coherente, pero no necesariamente útil, seguro o alineado con la intención del usuario.

Por eso se usan técnicas de **alineación**, entre ellas:

- **supervised fine-tuning** con ejemplos de instrucciones y respuestas,
- **RLHF** (*Reinforcement Learning from Human Feedback*),
- filtros de seguridad y políticas de uso.

El objetivo es favorecer respuestas:

- más útiles,
- menos tóxicas,
- más fieles a la intención de la consulta.

---

## 9. Inferencia: cómo genera texto

Durante uso real, el modelo:

1. recibe un prompt,
2. calcula una distribución de probabilidad sobre el siguiente token,
3. selecciona uno según una estrategia de decodificación,
4. repite el proceso hasta detenerse.

### Estrategias comunes

- **Greedy decoding**: elige siempre el token más probable.
- **Sampling**: introduce variedad.
- **Top-k / top-p**: limita el muestreo a un subconjunto plausible.
- **Temperature**: controla creatividad frente a determinismo.

---

## 10. Capacidades y límites

### Capacidades

- generación de texto,
- resumen,
- traducción,
- clasificación,
- respuesta a preguntas,
- apoyo a programación,
- extracción de información.

### Limitaciones

1. **Alucinaciones**: puede producir respuestas falsas pero plausibles.
2. **Sesgos**: aprende sesgos presentes en los datos.
3. **Dependencia del prompt**: pequeños cambios en la instrucción alteran el resultado.
4. **Costo computacional**: entrenamiento e inferencia son caros.
5. **No garantiza razonamiento formal**: puede imitar razonamiento sin hacerlo de forma robusta.

Un LLM modela **patrones estadísticos del lenguaje**, no una comprensión perfecta del mundo.

---

## 11. Relación con otros conceptos del curso

Los LLMs conectan varios temas de NLP:

- **n-grams**: punto de partida histórico.
- **embeddings**: base para representar tokens en espacios continuos.
- **redes neuronales**: fundamento de aprendizaje profundo.
- **evaluación**: perplexity para modelado; exactitud, F1 u otras métricas para tareas downstream.

En ese sentido, los LLMs no reemplazan los temas anteriores: los integran y los escalan.

---

## 12. Conclusión

El resumen original explica bien el origen probabilístico del modelado de lenguaje, MLE, n-grams, perplexity y la transición hacia modelos neuronales. Para completarlo, es fundamental añadir que los LLMs modernos:

- usan **tokenización subword**,
- se apoyan en **Transformers** y **self-attention**,
- aprenden mediante **pretraining autorregresivo**,
- mejoran con **escala**,
- suelen refinarse con **fine-tuning** e **instruction tuning**,
- requieren procesos de **alineación** para ser útiles y seguros.

En síntesis, un LLM es un modelo probabilístico neuronal a gran escala que aprende a predecir texto, pero cuya potencia práctica proviene de la arquitectura Transformer, la disponibilidad masiva de datos y la adaptación posterior a instrucciones humanas.
