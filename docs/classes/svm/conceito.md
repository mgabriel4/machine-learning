# Support Vector Machine (SVM)

Support Vector Machine (SVM) é um algoritmo supervisionado usado para classificação e regressão, cujo objetivo principal é encontrar o hiperplano que melhor separa os dados.
Esse “melhor” hiperplano é aquele que maximiza a margem, ou seja, a distância entre o hiperplano e os pontos mais próximos de cada classe — os famosos support vectors.

## Intuição do SVM

Imagine duas classes de pontos num plano 2D.
Existem infinitas linhas que separam essas classes, mas apenas uma maximiza a distância até os pontos mais próximos.
Essa linha é o hiperplano ótimo.

* Maior margem = melhor generalização

* Os únicos pontos que importam para definir o hiperplano são os support vectors

* O SVM é robusto contra overfitting em muitos casos

## Conceitos Fundamentais

1. **Hiperplano**: A fronteira de decisão que separa as classes.
    * Em 2D → uma linha
    * Em 3D → um plano
    * Em n-dimensões → hiperplano

2. **Margem**: A distância entre o hiperplano e os support vectors. O SVM busca maximizar essa margem.

3. **Support Vectors**: Pontos críticos que ficam na borda da margem. Eles são responsáveis por definir o hiperplano ótimo.

4. **SVM Linear vs. Não Linear**
    * Linear: Funciona quando os dados são separáveis por uma linha (2D) ou plano (nD).
    * Não Linear: Quando os dados não são separáveis linearmente, usamos kernels.

## Kernel Trick

Um dos pilares do SVM é a kernel trick, que permite projetar dados não lineares em um espaço de maior dimensão sem precisar calcular explicitamente essa transformação.

| Kernel             | Quando usar                                                |
| ------------------ | ---------------------------------------------------------- |
| **Linear**         | Dados linearmente separáveis, muito rápido e interpretável |
| **Polinomial**     | Relações não lineares suaves                               |
| **RBF (Gaussian)** | Kernel mais usado; captura relações complexas              |
| **Sigmoid**        | Similar a redes neurais, menos comum                       |

## Parâmetros Importantes

1. **C (Regularização)**: Controla o trade-off entre:
    * Margem larga
    * Erros aceitáveis na classificação
    * C alto → modelo rígido (menos margem)
    * C baixo → modelo mais flexível (margem maior)

2. **Gamma (γ — usado no kernel RBF)**: Define o alcance da influência de um ponto.
    * γ alto → fronteira complexa (risco de overfitting)
    * γ baixo → fronteira suave

## SVM para Regressão (SVR)

A versão para regressão busca aproximar uma função dentro de um intervalo de tolerância ε.
Objetivo: encontrar uma função o mais plana possível, penalizando desvios acima de ε.

## Vantagens

* Funciona muito bem com margens claras entre classes

* Eficiente em alta dimensionalidade

* Robusto ao overfitting quando bem ajustado

* Kernel trick permite modelar relações complexas

## Desvantagens

* Pode ser muito lento em datasets grandes

* Alta sensibilidade à escolha de hiperparâmetros

* Difícil interpretar fronteiras não lineares

* Requer normalização dos dados

## Quando usar SVM?

Use SVM quando:

* O dataset não é gigantesco

* Você quer alta acurácia

* Há separabilidade clara (linear ou não linear)

* Outliers são controlados

Evite SVM quando:

* O dataset tem milhões de linhas

* Você precisa explicar facilmente o modelo

## Pipeline típico com SVM (em qualquer linguagem)

1. Normalizar/Padronizar os dados

2. Escolher kernel

3. Ajustar hiperparâmetros (C, gamma, degree, etc.)

4. Treinar o modelo

5. Avaliar com validação cruzada

6. Interpretar (se usado kernel linear)
