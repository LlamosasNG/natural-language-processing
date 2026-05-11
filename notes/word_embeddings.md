# Word Embeddings: Guía de Estudio

> **Traducción y explicación del material original**  
> Fecha del documento original: martes, 3 de marzo de 2026

---

## 1. Representaciones Distribuidas

### 1.1 Similitud Distribucional

**Idea central:** El significado de una palabra puede entenderse a partir del **contexto** en el que aparece.

Esto también se conoce como **connotación**: el significado está definido por el contexto. Se opone a la **denotación**, que es el significado literal de una palabra.

**Ejemplo:**
> "NLP rocks" (El PNL es increíble)

- Significado literal de "rocks": "rocas/piedras"
- Significado contextual: algo bueno y moderno

### 1.2 Hipótesis Distribucional

**En lingüística:** Las palabras que ocurren en contextos similares tienen significados similares.

**Ejemplo:** Las palabras "dog" (perro) y "cat" (gato) aparecen en contextos similares. Según la hipótesis distribucional, debe haber una fuerte similitud entre sus significados.

**Representación vectorial:** Si dos palabras frecuentemente ocurren en contextos similares, sus vectores de representación deben estar cercanos entre sí en el espacio vectorial.

### 1.3 Representación Distribucional

Esquemas de representación obtenidos basándose en la **distribución de palabras** a partir del contexto en que aparecen.

**Características:**
- Basados en la hipótesis distribucional
- La propiedad distribucional se induce del contexto (vicindad textual)
- Usan vectores de **alta dimensionalidad**
- Se obtienen de una **matriz de co-ocurrencia** que captura la relación palabra-contexto
- La dimensión de esta matriz es igual al tamaño del vocabulario del corpus

**Ejemplos de esquemas:**
- Bag of Words (Bolsa de palabras)
- Bag of n-grams (Bolsa de n-gramas)
- TF-IDF

### 1.4 Representación Distribuida (Distributed Representation)

Concepto relacionado, también basado en la hipótesis distribucional, pero con una mejora clave:

| Representación Distribucional | Representación Distribuida |
|------------------------------|---------------------------|
| Vectores de **muy alta dimensión** | Vectores de **baja dimensión** |
| **Dispersos** (muchos ceros) | **Densos** (casi ningún cero) |
| Computacionalmente ineficientes | Computacionalmente eficientes |

**Ventaja:** Comprime significativamente la dimensionalidad, facilitando el aprendizaje automático.

---

## 2. Semántica Vectorial

**Definición:** Conjunto de métodos de PLN (Procesamiento de Lenguaje Natural) que buscan aprender representaciones de palabras basadas en propiedades distribucionales de palabras en un corpus grande.

---

## 3. Word Embeddings: El Concepto

### Origen del Término

El término "word embeddings" fue popularizado por **Yoshua Bengio et al. en 2003** para describir el mapeo de palabras en un **espacio vectorial continuo y denso**, donde se preservan las relaciones semánticas.

**Metáfora geológica:** Las palabras están "incrustadas" (embedded) como fósiles en un espacio de alta dimensionalidad.

**Referencia:** Bengio, Yoshua, et al. "A neural probabilistic language model." Journal of Machine Learning Research 3 (2003): 1137–1155.

### Metáfora de los Fósiles

| Característica | Explicación |
|---------------|-------------|
| **Preservación de estructura** | Así como los fósiles mantienen su forma mientras están incrustados en roca, los documentos deben mantener sus relaciones funcionales mientras están incrustados en el espacio vectorial |
| **Integración** | El objeto incrustado se vuelve parte integral del marco computacional más grande mientras retiene su identidad |
| **Accesibilidad** | Una vez incrustados, los documentos pueden compararse, agruparse y analizarse dentro de su nuevo contexto matemático |

---

## 4. Modelos Principales de Word Embeddings

### 4.1 Word2Vec (Predictivo) - Google

Utiliza dos arquitecturas principales:

#### CBOW (Continuous Bag of Words)
- **Función:** Predice una palabra objetivo basándose en el contexto
- **Ejemplo:** "The [?] barked at the mailman" → Predice "dog"

#### Skip-gram
- **Función:** Predice el contexto basándose en una palabra objetivo
- **Ejemplo:** "Barked" → Predice ["The", "Dog", "At", "mailman"]

### 4.2 GloVe (Global Vectors) - Stanford

- Enfoque: Mira el **panorama completo**
- Analiza la **matriz global de co-ocurrencia** de todo el dataset
- Combina ventajas de métodos basados en conteo y predictivos

---

## 5. Evolución: De Estático a Contextual

### Embeddings Estáticos (Antiguos)

La palabra **"Bank"** tiene **un solo vector**, sin importar el contexto:
- 🏦 River bank (orilla del río)
- 💰 Piggy bank (alcancía)

**Problema:** No captura la polisemia (múltiples significados).

### Embeddings Contextuales (Modernos - Transformers)

Modelos como **BERT** o **Gemini** generan un **vector diferente** para "Bank" dependiendo de las palabras que lo rodean.

**Ventaja:** Captura el significado contextual preciso.

---

## 6. Tendencias 2026

### 6.1 Embeddings Multimodales

- Mapean **imágenes y texto** en el mismo espacio vectorial
- **Aplicación:** Buscar "zapatos rojos" y encontrar una imagen real de zapatos rojos
- **Modelos:** CLIP, DALL-E, etc.

### 6.2 Embeddings Matryoshka

- Modelos modernos (como **text-embedding-3-large** de OpenAI) permiten "encoger" un vector
- **Ejemplo:** De 3072 dimensiones → 256 dimensiones
- **Beneficio:** Ahorro masivo de memoria sin perder mucha precisión

---

## 7. Representación de Documentos

### Enfoque Simple

1. Dividir el texto en palabras constituyentes
2. Obtener embeddings individuales para cada palabra
3. Combinarlos para formar la representación del texto

### Métodos de Combinación

| Método | Descripción |
|--------|-------------|
| **Suma** | Sumar todos los vectores de palabras |
| **Promedio** | Calcular el vector promedio |
| **Otros** | Máximo, mínimo, ponderado por TF-IDF, etc. |

---

## 8. Resumen Visual

```
┌─────────────────────────────────────────────────────────────┐
│                    EVOLUCIÓN DE EMBEDDINGS                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  2003: Bengio → Word Embeddings (concepto)                  │
│         ↓                                                   │
│  2013: Word2Vec (Google) - CBOW & Skip-gram                 │
│         ↓                                                   │
│  2014: GloVe (Stanford) - Co-ocurrencia global              │
│         ↓                                                   │
│  2018+: BERT/Transformers - Contextuales                    │
│         ↓                                                   │
│  2026: Multimodal + Matryoshka                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Conceptos Clave para Recordar

| Término | Definición |
|---------|------------|
| **Hipótesis distribucional** | Palabras en contextos similares = significados similares |
| **Co-ocurrencia** | Frecuencia con que dos palabras aparecen juntas |
| **Espacio vectorial** | Representación matemática donde palabras = vectores |
| **Densidad** | Vectores con pocos ceros (eficientes) |
| **Dispersión** | Vectores con muchos ceros (ineficientes) |
| **Contextual** | El embedding cambia según el contexto de la palabra |

---

## 10. Aplicaciones Prácticas

1. **Búsqueda semántica:** Encontrar documentos por significado, no por palabras clave
2. **Similitud de palabras:** Calcular qué tan similares son dos palabras
3. **Analogías:** "rey" - "hombre" + "mujer" = "reina"
4. **Clasificación de texto:** Categorizar documentos automáticamente
5. **Traducción automática:** Mapear palabras entre idiomas
6. **Sistemas de recomendación:** Sugerir contenido similar

---

## Glosario Español-Inglés

| Español | Inglés |
|---------|--------|
| Incrustación de palabras | Word Embeddings |
| Bolsa de palabras | Bag of Words |
| Co-ocurrencia | Co-occurrence |
| Espacio vectorial | Vector Space |
| Semántica vectorial | Vector Semantics |
| Aprendizaje distribuido | Distributed Learning |
| Corpus | Corpus |
| Dimensionalidad | Dimensionality |

---

> **Nota de estudio:** Los word embeddings son fundamentales en PLN moderno. Comprender la diferencia entre representaciones **distribucionales** (altas dimensiones, dispersas) y **distribuidas** (bajas dimensiones, densas) es clave para entender la evolución hacia modelos como BERT y los embeddings contextuales.
