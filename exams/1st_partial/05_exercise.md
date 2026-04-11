# Ejercicio 5

### Toy Corpus

> **D1:** "cats chase mice"
> **D2:** "dogs chase cats"
> **D3:** "mice and cats play"

### Vocabulary

|V| = { cats = 0, chase = 1, mice = 2, dogs = 3, and = 4, play = 5 }

---

## Bag of Words

### Vectors

|        | cats | chase | mice | dogs | and | play |
|--------|:----:|:-----:|:----:|:----:|:---:|:----:|
| **D1** | 1    | 1     | 1    | 0    | 0   | 0    |
| **D3** | 1    | 0     | 1    | 0    | 1   | 1    |

- **D1** → `[1, 1, 1, 0, 0, 0]`
- **D3** → `[1, 0, 1, 0, 1, 1]`

---

## TF-IDF

### Cálculo de df(t)

| Término | df(t) | Documentos donde aparece |
|---------|:-----:|--------------------------|
| cats    | 3     | D1, D2, D3               |
| chase   | 2     | D1, D2                   |
| mice    | 2     | D1, D3                   |
| dogs    | 1     | D2                       |
| and     | 1     | D3                       |
| play    | 1     | D3                       |

### Cálculo de IDF

$$IDF(t) = \log_{10}\left(\frac{N}{df(t)}\right) \quad \text{donde } N = 3$$

| Término | Fórmula               | Resultado    |
|---------|------------------------|--------------|
| cats    | log(3 / 3) = log(1)   | **0**        |
| chase   | log(3 / 2) = log(1.5) | **0.1761**   |
| mice    | log(3 / 2) = log(1.5) | **0.1761**   |
| dogs    | log(3 / 1) = log(3)   | **0.4771**   |
| and     | log(3 / 1) = log(3)   | **0.4771**   |
| play    | log(3 / 1) = log(3)   | **0.4771**   |

---

### Document 1: "cats chase mice"

| Término | TF | IDF    | TF-IDF = TF × IDF |
|---------|:--:|:------:|:------------------:|
| cats    | 1  | 0      | **0**              |
| chase   | 1  | 0.1761 | **0.1761**         |
| mice    | 1  | 0.1761 | **0.1761**         |
| dogs    | 0  | 0.4771 | **0**              |
| and     | 0  | 0.4771 | **0**              |
| play    | 0  | 0.4771 | **0**              |

**Vector TF-IDF de D1:**

|  cats  | chase  |  mice  |  dogs  |  and   |  play  |
|:------:|:------:|:------:|:------:|:------:|:------:|
|   0    | 0.1761 | 0.1761 |   0    |   0    |   0    |

---

### Document 2: "dogs chase cats"

| Término | TF | IDF    | TF-IDF = TF × IDF |
|---------|:--:|:------:|:------------------:|
| cats    | 1  | 0      | **0**              |
| chase   | 1  | 0.1761 | **0.1761**         |
| mice    | 0  | 0.1761 | **0**              |
| dogs    | 1  | 0.4771 | **0.4771**         |
| and     | 0  | 0.4771 | **0**              |
| play    | 0  | 0.4771 | **0**              |

**Vector TF-IDF de D2:**

|  cats  | chase  |  mice  |  dogs  |  and   |  play  |
|:------:|:------:|:------:|:------:|:------:|:------:|
|   0    | 0.1761 |   0    | 0.4771 |   0    |   0    |