# Evaluación de Clasificadores (Classifier Evaluation)

Este documento explica cómo medir el rendimiento de un modelo de clasificación de texto comparando las predicciones del sistema con las etiquetas reales (establecidas por humanos), conocidas como **etiquetas de oro (gold labels)**.

---

## 1. La Matriz de Confusión
La matriz de confusión es una tabla que permite visualizar el desempeño de un algoritmo. Cruza las etiquetas reales (gold) con las etiquetas predichas por el sistema.

### Definiciones Clave:
| Término | Nombre | Descripción |
| :--- | :--- | :--- |
| **TP** | Verdadero Positivo | El sistema dijo "Positivo" y era "Positivo". |
| **FP** | Falso Positivo | El sistema dijo "Positivo" pero era "Negativo". (Error Tipo I) |
| **FN** | Falso Negativo | El sistema dijo "Negativo" pero era "Positivo". (Error Tipo II) |
| **TN** | Verdadero Negativo | El sistema dijo "Negativo" y era "Negativo". |

---

## 2. Métricas de Rendimiento

### Accuracy (Exactitud)
Es el porcentaje total de predicciones correctas.
$$\text{Accuracy} = \frac{TP + TN}{TP + FP + TN + FN}$$
*   **Limitación:** No es confiable cuando las clases están **desbalanceadas** (ej. si el 99% de los correos no son spam, un modelo que siempre diga "no spam" tendrá 99% de accuracy pero no detectará ningún spam).

### Precision (Precisión)
Mide qué tan confiable es el sistema cuando dice que algo es positivo.
$$\text{Precision} = \frac{TP}{TP + FP}$$
*   **Pregunta que responde:** "De todo lo que el sistema marcó como positivo, ¿cuánto es realmente positivo?"

### Recall (Exhaustividad / Sensibilidad)
Mide la capacidad del sistema para encontrar todos los casos positivos.
$$\text{Recall} = \frac{TP}{TP + FN}$$
*   **Pregunta que responde:** "De todos los positivos que existen realmente, ¿cuántos logró detectar el sistema?"

---

## 3. Medida F (F-Measure)
Es la media armónica ponderada entre la precisión y el recall. Se utiliza para encontrar un equilibrio entre ambas métricas.

### Fórmula General ($F_\beta$):
$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}$$

*   **$\beta > 1$:** Da más importancia al **Recall** (útil en medicina, donde es peor omitir un enfermo que tener un falso positivo).
*   **$\beta < 1$:** Da más importancia a la **Precision** (útil en filtros de spam, donde es peor borrar un correo importante que dejar pasar un spam).
*   **$\beta = 1$ (F1-Score):** Balance equitativo entre precisión y recall.
    $$\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 4. Complemento: El Dilema Precision vs. Recall
Existe un compromiso (*trade-off*) entre estas dos métricas. Si intentas aumentar la precisión (siendo muy selectivo), probablemente bajes el recall (ignoras casos positivos dudosos). Si intentas aumentar el recall (siendo muy inclusivo), probablemente bajes la precisión (incluyes más falsos positivos).

**Ejemplo práctico:**
- **Filtro de Spam:** Preferimos alta **Precisión**. No importa si llega un spam a la bandeja de entrada (FN), pero es crítico que el sistema no mueva un correo real a la carpeta de spam (FP).
- **Detección de Cáncer:** Preferimos alto **Recall**. Es preferible hacer una prueba extra a alguien sano (FP) que decirle a alguien enfermo que está sano y dejarlo sin tratamiento (FN).
