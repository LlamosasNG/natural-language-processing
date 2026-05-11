# Extracción de Información - Análisis y Traducción

**Fecha del documento original:** Martes, 10 de marzo de 2026

**Fuente:** Maas, A., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis.

---

## 1. Introducción al Análisis de Sentimientos

El documento presenta un sistema de **clasificación de sentimientos** basado en aprendizaje automático, utilizando reseñas de películas como ejemplo.

### Ejemplos de Reseñas

**Reseña 1 (Positiva):**
> "This is a poem on film, wonderfully presented and photographed with sensitive artistry. It captures the atmosphere of the time and place perfectly..."

**Reseña 2 (Negativa):**
> "The title alone (along with the poster) is enough to give away 'The Projected Man' as an obvious rip-off of 'The Fly'..."

---

## 2. Extracción de Características (Features)

El sistema extrae las siguientes características de cada texto:

| Característica | Descripción |
|----------------|-------------|
| **X1 - Puntuación** | Conteo de signos (?????? = Negativo, !!!!!!!!! = Positivo) |
| **X2 - Léxico Positivo** | Palabras positivas: [poem, wonderfully, artistry, perfectly, love, unique, deliciously, beautiful, happy, masterpiece] |
| **X3 - Léxico Negativo** | Palabras negativas: [rip-off, acceptable, horribly, wrong, pass, isn't] |
| **X4 - Log(longitud)** | Logaritmo de la longitud del texto |

---

## 3. Modelo de Clasificación

### Función de Decisión

El modelo calcula una **suma ponderada de características**:

```
z = (Σ wᵢxᵢ) + b = w·x + b
```

Donde:
- `w` = pesos aprendidos
- `x` = vector de características
- `b` = sesgo (bias)

### Función Sigmoide

La función sigmoide transforma el valor `z` en una probabilidad entre 0 y 1:

```
σ(z) = 1 / (1 + e⁻ᶻ)
```

Esto permite modelar el problema como:
- **Positivo (y=1):** P(y=1) = σ(z) = σ(w·x + b)
- **Negativo (y=0):** P(y=0) = 1 - σ(z) = 1 - σ(w·x + b)

La predicción del modelo es:
```
ŷ = p(y|x) = σ(z) = σ(w·x + b)
```

---

## 4. Función de Pérdida: Entropía Cruzada

### Definición

Necesitamos una función de pérdida que mida qué tan cerca está la salida estimada (ŷ) de la salida correcta (y, que es 0 o 1).

Dado que hay solo dos resultados discretos (1 o 0), esto sigue una **distribución Bernoulli**:

```
p(y|x) = ŷʸ(1-ŷ)¹⁻ʸ
```

### Derivación de la Función de Pérdida

Tomando logaritmo (maximizar la probabilidad es equivalente a maximizar su logaritmo):

```
log p(y|x) = log(ŷʸ(1-ŷ)¹⁻ʸ)
           = y log ŷ + (1-y) log(1-ŷ)
```

Esto describe una **verosimilitud logarítmica** que debe maximizarse. Para convertirla en una función de pérdida (que debemos minimizar), cambiamos el signo:

```
L_CE(ŷ,y) = -log p(y|x)
          = -(y log ŷ + (1-y) log(1-ŷ))
          = -(y log σ(w·x + b) + (1-y) log(1-σ(w·x + b)))
```

### Ejemplos de Cálculo

| Caso | ŷ | y | Pérdida L_CE | Interpretación |
|------|---|---|--------------|----------------|
| Cerca de correcto | 0.7 | 1 | ≈ 0.36 | Pérdida baja |
| Modelo confundido | 0.7 | 0 | ≈ 1.2 | Pérdida alta |

---

## 5. Minimización de la Función de Pérdida

### Objetivo

Sea **Θ = {w, b}** el conjunto de parámetros a aprender. El objetivo es encontrar los pesos que minimizan la función de pérdida, promediada sobre todos los ejemplos:

```
Θ* = argmin_Θ (1/N) Σ L_CE(ŷ⁽ⁱ⁾, y⁽ⁱ⁾)
```

### Descenso de Gradiente (Gradient Descent)

El algoritmo de optimización actualiza los parámetros iterativamente:

```
Θ ← Θ - η ∇_Θ L_CE
```

Donde:
- **η** = tasa de aprendizaje (learning rate), un hiperparámetro
- **∇_Θ L_CE** = gradiente de la función de pérdida respecto a los parámetros

### Criterios de Parada

1. **Número fijo de iteraciones**
2. **Umbral** (en Θ, L_CE, ΔΘ, ΔL_CE, etc.)
3. **Monitoreo** (early stopping, validación)

---

## 6. Nombre del Modelo

Este modelo también se conoce como **Clasificador de Regresión Logística** (Logistic Regression Classifier).

---

## 7. Conceptos Clave Resumidos

| Concepto | Descripción |
|----------|-------------|
| **Extracción de características** | Convertir texto en vectores numéricos |
| **Función sigmoide** | Mapea cualquier valor a (0, 1) para interpretación como probabilidad |
| **Entropía cruzada** | Mide la diferencia entre distribución predicha y real |
| **Descenso de gradiente** | Algoritmo iterativo para encontrar mínimos de la función de pérdida |
| **Regresión logística** | Modelo lineal para clasificación binaria |

---

## 8. Diagrama del Flujo del Modelo

```
Texto → Extracción de Features → Vector x → w·x + b → Sigmoide → ŷ → Comparar con y → Calcular Pérdida → Actualizar pesos
```

---

## Referencias

Maas, A., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011, June). Learning word vectors for sentiment analysis. In *Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies* (pp. 142-150).
