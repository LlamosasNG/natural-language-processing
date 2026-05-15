# Transformers: guía de estudio

> Análisis y ampliación del resumen `docs/sources/summaries/transformers.md`
> Tema central: arquitectura Transformer, self-attention y su relación con los LLMs

---

## 1. Idea central

Los **Transformers** son una arquitectura de redes neuronales diseñada para procesar secuencias sin depender de recurrencia paso a paso. Su innovación principal es el mecanismo de **self-attention**, que permite que cada token compare su representación con todos los demás tokens del contexto y construya una versión contextualizada de sí mismo.

La intuición básica es:

- cada palabra formula una "consulta" sobre qué información necesita;
- compara esa consulta con el resto de palabras;
- asigna más peso a las palabras relevantes;
- combina esa información para actualizar su representación.

Esto resuelve un problema clásico de RNNs y LSTMs: en modelos estrictamente secuenciales, la información lejana debe propagarse por muchos pasos y puede degradarse. En un Transformer, dos tokens distantes pueden interactuar directamente en una sola capa.

---

## 2. ¿Qué problema resuelven?

Antes de Transformers, gran parte del NLP profundo dependía de **RNNs**, **LSTMs** y **GRUs**. Esos modelos fueron útiles, pero tenían varias limitaciones:

- el procesamiento era secuencial y por tanto lento;
- paralelizar el entrenamiento era más difícil;
- modelar dependencias largas seguía siendo costoso;
- el contexto se comprimía de forma indirecta en estados ocultos.

El Transformer cambia el enfoque:

- elimina la recurrencia como mecanismo principal;
- procesa todas las posiciones de una secuencia en paralelo durante entrenamiento;
- modela relaciones globales mediante atención;
- construye contexto de forma explícita.

Por eso los Transformers se volvieron la base de los modelos modernos de traducción, resumen, clasificación y, sobre todo, de los **LLMs**.

---

## 3. Representación de entrada

Antes de aplicar atención, el texto debe convertirse en vectores.

### 3.1 Tokenización

El texto se divide en **tokens**, que suelen ser subpalabras en lugar de palabras completas. Esto ayuda a:

- reducir el tamaño del vocabulario;
- manejar palabras raras;
- reutilizar fragmentos frecuentes como prefijos, sufijos o raíces.

### 3.2 Embeddings

Cada token se transforma en un vector denso. Ese vector captura información aprendida durante entrenamiento sobre cómo aparece el token en distintos contextos.

### 3.3 Información posicional

Como el Transformer no procesa secuencias de izquierda a derecha por diseño recurrente, necesita una forma explícita de saber el orden de los tokens. Para eso se añaden **positional encodings** o **positional embeddings** a los embeddings de entrada.

Sin información posicional, una secuencia como:

- "el perro mordió al hombre"
- "el hombre mordió al perro"

tendría los mismos tokens, pero el modelo no distinguiría bien el orden.

---

## 4. Self-attention

La auto-atención es el núcleo de la arquitectura.

Cada token genera tres vectores:

- **Query (Q)**: qué información está buscando;
- **Key (K)**: qué información ofrece;
- **Value (V)**: qué contenido aporta.

Estos vectores se obtienen mediante proyecciones lineales aprendidas:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

donde \(X\) es la matriz de representaciones de entrada.

### 4.1 Cálculo de la atención

La similitud entre una query y todas las keys se calcula con productos punto:

$$
\text{score}(Q, K) = QK^T
$$

Luego se escala y normaliza con softmax:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

La división por \(\sqrt{d_k}\) evita que los productos punto crezcan demasiado y vuelvan inestable el softmax.

### 4.2 Interpretación intuitiva

Para un token dado:

1. compara su query con las keys de todos los tokens;
2. obtiene pesos de importancia;
3. usa esos pesos para combinar los values;
4. produce una representación contextualizada.

Así, una palabra ambigua puede desambiguarse según el contexto. Por ejemplo, "banco" puede asociarse con "dinero" o con "río" según los tokens cercanos que reciban más atención.

---

## 5. Multi-head attention

El resumen original menciona self-attention, pero conviene añadir una idea clave: en la práctica se usa **multi-head attention**.

En lugar de calcular una sola atención, el modelo calcula varias atenciones en paralelo, llamadas **heads**. Cada cabeza puede especializarse en patrones distintos, por ejemplo:

- relaciones sintácticas;
- concordancia entre sujeto y verbo;
- referencias anafóricas;
- dependencias semánticas más globales.

Formalmente:

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

Luego las cabezas se concatenan y se proyectan:

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
$$

La ventaja no es solo tener "más atención", sino aprender distintos tipos de relaciones al mismo tiempo.

---

## 6. Bloque Transformer

Un bloque Transformer estándar contiene:

1. capa de atención;
2. conexión residual;
3. layer normalization;
4. red feed-forward por posición;
5. otra conexión residual;
6. otra layer normalization.

Una forma esquemática es:

```text
Input
  |
Multi-Head Attention
  |
Add + LayerNorm
  |
Feed-Forward Network
  |
Add + LayerNorm
  |
Output
```

### 6.1 Conexiones residuales

Las conexiones residuales ayudan a:

- estabilizar el entrenamiento;
- facilitar el flujo de gradientes;
- permitir redes profundas sin degradación severa.

### 6.2 Layer normalization

La normalización ayuda a mantener escalas numéricas estables y hace más robusto el entrenamiento.

### 6.3 Feed-forward network

Después de la atención, cada posición pasa por una red totalmente conectada:

$$
FFN(x) = \max(0, W_1x + b_1)W_2 + b_2
$$

Se aplica la misma transformación a cada posición, de forma independiente, pero sobre representaciones ya contextualizadas por la atención.

---

## 7. Encoder y decoder

No todos los Transformers se usan igual. Existen dos configuraciones importantes.

### 7.1 Encoder

El **encoder** procesa toda la secuencia de entrada y genera representaciones contextualizadas. Es útil en tareas de comprensión, como:

- clasificación de texto;
- análisis de sentimiento;
- etiquetado;
- búsqueda semántica.

Modelos como **BERT** se basan principalmente en encoders.

### 7.2 Decoder

El **decoder** genera texto de manera autorregresiva: predice el siguiente token a partir de los anteriores. Para esto usa **masked self-attention**, que impide ver tokens futuros.

Modelos como **GPT** son esencialmente Transformers **decoder-only**.

### 7.3 Encoder-decoder

En tareas de transformación de secuencias, como traducción automática, se usa la versión completa:

- un encoder procesa la entrada;
- un decoder genera la salida;
- el decoder además usa **cross-attention** para atender a la representación del encoder.

Esta variante fue central en traducción neuronal y en modelos como T5.

---

## 8. Masked attention y generación autorregresiva

En un LLM generativo, el modelo no puede mirar el futuro durante generación. Por eso se usa una máscara causal que obliga a cada posición \(t\) a atender solo a:

- los tokens anteriores;
- el token actual;
- nunca a los tokens posteriores.

Esto mantiene coherencia con el objetivo probabilístico:

$$
P(w_1^n) = \prod_{t=1}^{n} P(w_t \mid w_1^{t-1})
$$

Durante entrenamiento, muchas posiciones pueden procesarse en paralelo porque el texto completo ya está disponible y solo se aplica la máscara. Durante inferencia, la generación sigue siendo secuencial: un token nuevo depende de los ya generados.

---

## 9. Relación con el entrenamiento de LLMs

Los Transformers no cambian el objetivo probabilístico del modelado de lenguaje. Lo que cambian es la forma de calcular la distribución del siguiente token.

El flujo general es:

1. recibir una secuencia de tokens;
2. convertirla en embeddings con información posicional;
3. pasarla por varios bloques Transformer;
4. obtener logits sobre el vocabulario;
5. aplicar softmax para producir probabilidades;
6. optimizar con cross-entropy respecto al siguiente token real.

Es decir, el Transformer es la **arquitectura**; el modelado del siguiente token sigue siendo el **objetivo de aprendizaje**.

---

## 10. Muestreo y estrategias de decodificación

El resumen original acierta al separar dos conceptos:

- la arquitectura produce una distribución de probabilidad;
- el muestreo decide cómo elegir un token concreto.

Algunas estrategias comunes son:

### 10.1 Greedy decoding

Selecciona siempre el token más probable.

Ventajas:

- simple;
- determinista.

Desventajas:

- puede ser repetitivo;
- se atasca en continuaciones demasiado previsibles.

### 10.2 Muestreo multinomial

Elige aleatoriamente según la distribución predicha. Introduce diversidad, pero puede producir incoherencias si se asigna demasiada masa a opciones débiles.

### 10.3 Temperatura

Modifica la concentración de probabilidades:

$$
P_T(w) \propto \exp\left(\frac{s_w}{T}\right)
$$

- \(T < 1\): salida más conservadora;
- \(T > 1\): salida más diversa y riesgosa.

### 10.4 Top-k y top-p

- **Top-k**: restringe la elección a los \(k\) tokens más probables.
- **Top-p** o **nucleus sampling**: restringe la elección al conjunto mínimo de tokens cuya probabilidad acumulada supera un umbral \(p\).

Estas técnicas buscan un equilibrio entre coherencia y variedad.

---

## 11. ¿Por qué los Transformers fueron tan influyentes?

Su impacto no se debe a una sola ventaja, sino a la combinación de varias:

- entrenan mejor en hardware paralelo;
- escalan bien con más datos y más parámetros;
- capturan dependencias largas con mayor facilidad;
- sirven como base común para tareas de comprensión y generación;
- permiten transfer learning efectivo mediante pretraining y fine-tuning.

En la práctica, los Transformers se convirtieron en la arquitectura dominante porque ofrecieron mejores resultados y una ruta clara de escalamiento.

---

## 12. Limitaciones importantes

Aunque los Transformers son muy potentes, no resuelven todos los problemas.

### 12.1 Costo computacional

La self-attention completa tiene costo cuadrático respecto a la longitud de la secuencia:

$$
O(n^2)
$$

Esto vuelve costoso trabajar con contextos muy largos.

### 12.2 Ventana de contexto finita

Aunque pueden modelar relaciones largas dentro de una secuencia, siguen limitados por una longitud máxima de contexto.

### 12.3 Comprensión no garantizada

Un Transformer puede producir texto muy convincente sin tener una comprensión robusta o verificable del mundo.

### 12.4 Sensibilidad al prompt y al muestreo

Pequeños cambios en la entrada o en los parámetros de generación pueden producir salidas muy distintas.

### 12.5 Aritmética y razonamiento exacto

Como en otros LLMs, aprender patrones textuales no equivale a ejecutar algoritmos exactos. Por eso pueden aparecer errores en:

- cálculo multi-paso;
- lógica larga;
- seguimiento estricto de restricciones.

---

## 13. Relación con otros temas del curso

Transformers conectan varios conceptos ya vistos en NLP:

- **embeddings**: son la base de las representaciones de entrada;
- **redes neuronales**: el bloque Transformer sigue siendo una red profunda entrenada por gradiente;
- **modelado de lenguaje**: conserva el objetivo probabilístico de predecir tokens;
- **clasificación**: puede adaptarse a tareas supervisadas;
- **LLMs**: los modelos modernos de gran escala suelen construirse sobre Transformers decoder-only.

Por eso entender Transformers ayuda a unir temas clásicos del curso con sistemas modernos como BERT, GPT y T5.

---

## 14. Conclusión

El resumen original presenta correctamente la intuición de self-attention, la estructura general del bloque Transformer y la relación entre Transformers y muestreo. Para completar el tema, es importante añadir que:

- la atención se formaliza con vectores **Q, K, V**;
- en la práctica se usa **multi-head attention**;
- el orden se incorpora con **información posicional**;
- existen variantes **encoder**, **decoder** y **encoder-decoder**;
- los LLMs generativos usan **masked self-attention**;
- la arquitectura mejora el modelado del contexto, pero no elimina límites de costo, contexto ni razonamiento exacto.

En síntesis, un Transformer es una arquitectura que convierte secuencias de tokens en representaciones contextuales mediante atención. Su capacidad para escalar, paralelizar el entrenamiento y modelar relaciones globales explica por qué se convirtió en la base técnica de los LLMs modernos.
