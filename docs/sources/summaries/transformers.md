# Transformers

## Contexto Previo

De los LLMs sabemos:
- Modelado de lenguaje = predecir el siguiente token
- Objetivo MLE / cross-entropy
- LMs neurales y representaciones contextuales (ELMo)
- Entrenamiento en corpus de texto masivos

**Problemas pendientes**:
- Modelos secuenciales son lentos
- Dependencias a largo plazo son difíciles
- Construcción de contexto es indirecta

Los Transformers resuelven estos problemas cambiando cómo se calcula el contexto.

## Auto-Atención (Self-Attention)

**Intuición**: La auto-atención permite a cada palabra:
- Mirar a todas las demás palabras
- Decidir cuáles importan
- Combinar su información

En lugar de:
- Comprimir todo en un estado oculto
- Pasar información paso a paso

**Modelo mental de un minuto**:
1. Preguntar: ¿Qué estoy buscando?
2. Comparar con todas las demás palabras
3. Tomar un promedio ponderado de lo que importa

**Consecuencia clave**:
- Cualquier palabra puede influir directamente en cualquier otra
- Dependencias a larga distancia son fáciles

## Bloques Transformer (Arquitectura)

Una capa de Transformer consiste en:
1. Auto-atención (Self-Attention)
2. Add & LayerNorm
3. Red feed-forward
4. Add & LayerNorm

**Red Feed-Forward**:
$$FFN(x) = \max(0, W_1 x + b_1)W_2 + b_2$$

- Agrega no linealidad
- Aumenta el poder expresivo
- Mismos pesos para todas las posiciones

**Arquitectura**:
```
Input embeddings
    |
Self-Attention
    |
Add + Norm
    |
Feed-Forward
    |
Add + Norm
    |
Output
```

Múltiples capas apiladas → Transformer

## ¿Por qué el nombre "Transformer"?

El nombre refleja lo que el modelo hace conceptualmente:
- Transforma una secuencia de representaciones de entrada
- En una secuencia de representaciones contextualizadas
- Usando transformaciones basadas en atención en cada capa

## Muestreo en LLMs y Transformers

### 1. ¿Qué es el muestreo?

Procedimiento usado durante la generación para convertir la distribución de probabilidad predicha por el modelo en un token real.

Un modelo de lenguaje (LLM o Transformer) define:
$$P(w_t | w_1^{t-1})$$

Dada la secuencia de tokens generados hasta ahora, ¿cuál es la probabilidad de cada posible siguiente token?

### 2. El muestreo es independiente de la arquitectura

Punto conceptual crucial:
- Los Transformers especifican cómo se calcula la distribución de probabilidad
- El muestreo especifica cómo seleccionamos un token de esa distribución
- Un Transformer calcula scores (logits) usando auto-atención
- Un RNN calcula scores usando un estado oculto recurrente
- Una vez producidos los scores, el muestreo funciona igual

### 3. Loop de generación unificado (LLMs + Transformers)

El proceso de generación es autorregresivo para todos los LLMs estándar:
1. Tomar los tokens generados hasta ahora: $w_1^{t-1}$
2. Calcular scores (logits) para todos los tokens del vocabulario
3. Aplicar softmax para obtener probabilidades
4. Samplear un token usando una estrategia de decodificación
5. Adjuntar el token y repetir

**Aclaración importante**: Los Transformers son paralelos durante el entrenamiento, pero secuenciales durante el muestreo.

### 4. Estrategias de Muestreo

#### 4.1 Decodificación Greedy
- Siempre elegir el token más probable
- Determinista
- A menudo repetitivo y plano

#### 4.2 Muestreo Multinomial
- Samplear proporcional a las probabilidades
- Más diverso
- Riesgo de incoherencia si se samplean tokens de baja probabilidad muy a menudo

#### 4.3 Escala de Temperatura
$$P_T(w) \propto \exp(\frac{s_w}{T})$$
- Baja temperatura → conservador, predecible
- Alta temperatura → creativo, riesgoso

#### 4.4 Muestreo Top-k y Top-p (Nucleus)
- Restringir el muestreo a un subconjunto de tokens probables
- Equilibrar coherencia y diversidad

**Idea clave**: Estas estrategias afectan el estilo y variabilidad, no el conocimiento subyacente del modelo.

### 5. ¿Qué cambió con los Transformers (y qué no)?

**Lo que no cambió**:
- El objetivo de probabilidad
- El loop de generación autorregresivo
- La necesidad de muestreo

**Lo que cambió**:
- La calidad de la distribución de probabilidad
- La capacidad de modelar dependencias a largo plazo
- La coherencia del texto generado

**Conclusión**: Los Transformers no cambian cómo muestreamos — solo de qué muestreamos.

## Limitaciones de LLMs y Transformers

### 1. Falta de verdadero razonamiento y comprensión

Los LLMs:
- Predicen el siguiente token usando regularidades estadísticas
- No manipulan símbolos usando reglas explícitas
- No "ejecutan" razonamiento lógico en el sentido tradicional

**Resultado**:
- Pueden producir explicaciones plausibles pero incorrectas
- Los errores a menudo emergen en razonamiento multi-paso largo

### 2. Aritmética y cómputo exacto

Limitación bien conocida respecto a operaciones aritméticas.

**Ejemplo**:
- Suma larga, multiplicación de enteros grandes
- Cálculos precisos multi-paso

**Por qué sucede**:
- Los LLMs aprenden aritmética como patrones en texto, no como algoritmos
- El entrenamiento optimiza verosimilitud, no corrección numérica
- Los resultados se degradan conforme los números crecen o el formato cambia

**Esto explica**:
- Aritmética pequeña puede ser correcta
- Redacción ligeramente diferente puede causar fracaso
- Los Transformers no implementan nativamente algoritmos aritméticos

### 3. Sensibilidad al prompt y muestreo

Due al muestreo:
- Pequeños cambios en temperatura o prompt pueden producir salidas muy diferentes
- El modelo puede contradecirse entre generaciones

Esto no es aleatoriedad en el conocimiento del modelo, sino en cómo se explora la masa de probabilidad durante el muestreo.

### 4. Límites de contexto y generalización

Incluso con auto-atención:
- Los modelos tienen ventanas de contexto finitas
- Información fuera de la ventana es inaccesible
- La "memoria" interna es implícita y frágil

**Además**:
- La generalización a entradas raras o adversariales es débil
- Los sesgos de los datos de entrenamiento se reflejan en las salidas