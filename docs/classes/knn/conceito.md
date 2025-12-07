# KNN (K-Nearest Neighbors)

O KNN é um algoritmo de aprendizado supervisionado usado para problemas de classificação e regressão. Ele é baseado na ideia de que objetos semelhantes estão próximos uns dos outros no espaço de características. É válido ressaltar que o KNN é bom para conjuntos de dados pequenos e médios, e é simples de entender e implementar.

## Funcionamento

1. **Escolha do K**: O primeiro passo é escolher o número de vizinhos (K) que serão considerados para a classificação ou regressão. Este passo é crucial, pois um valor muito pequeno pode tornar o modelo sensível ao ruído, enquanto um valor muito grande pode suavizar demais as fronteiras de decisão. Para que não haja uma escolha do K equivocada, faz-se a técnica de *validação cruzada*.

2. **Cálculo da Distância**: Para classificar um novo ponto, o algoritmo calcula a distância entre esse ponto e todos os pontos do conjunto de treinamento. As métricas de distâncias mais comuns são a Euclidiana, Manhattan e Minkowski.

3. **Identificação dos Vizinhos**: O algoritmo seleciona os K pontos mais próximos do conjunto de treinamento.

4. **Classificação ou Regressão**:
   * **Classificação**: O rótulo do novo ponto é determinado pelos rótulos frequentes dos K vizinhos. (Ou seja, a moda dos rótulos)
   * **Regressão**: O valor do novo ponto é determinado pela média (ou mediana) dos valores dos K vizinhos.

## Vantagens do KNN ✅

* Simplicidade: O KNN é fácil de entender e implementar.
* Flexibilidade: Pode ser usado para classificação e regressão.
* Não paramétrico: Não faz suposições sobre a distribuição dos dados.

## Desvantagens do KNN ❌

* Custo computacional: O KNN pode ser lento, especialmente com grandes conjuntos de dados, pois precisa calcular a distância de todos os pontos.
* Sensibilidade a ruídos: O algoritmo pode ser afetado por outliers e ruídos nos dados.
* Escolha do K: A escolha do valor de K pode impactar significativamente o desempenho do modelo.

### Métricas de Avaliação

As métricas de avaliação para KNN incluem:

* **Acurácia**: Proporção de previsões corretas em relação ao total de previsões.
* **Precisão**: Proporção de verdadeiros positivos em relação ao total de positivos previstos.
* **Revocação**: Proporção de verdadeiros positivos em relação ao total de positivos reais.
* **F1-Score**: Média harmônica entre precisão e revocação. Util quando há um desequilíbrio entre classes.
* **Matriz de Confusão**: Tabela que mostra o desempenho do modelo, detalhando verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos.
