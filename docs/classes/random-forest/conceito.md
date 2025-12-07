# Random Forest

O Random Forest é um algoritmo de aprendizado supervisionado que combina múltiplas árvores de decisão para melhorar a precisão e robustez das previsões. Ele é amplamente utilizado para tarefas de classificação e regressão devido à sua capacidade de lidar com grandes conjuntos de dados e alta dimensionalidade.

## Funcionamento

1. **Construção das Árvores**: O Random Forest constrói várias árvores de decisão a partir de diferentes subconjuntos aleatórios dos dados de treinamento. Cada árvore é treinada em uma amostra bootstrap (amostragem com reposição) dos dados.

2. **Seleção de Recursos**: Durante a construção de cada árvore, um subconjunto aleatório de recursos (features) é selecionado para determinar a melhor divisão em cada nó. Isso ajuda a reduzir a correlação entre as árvores e melhora a generalização do modelo.

3. **Agregação de Resultados**: Para fazer previsões, o Random Forest agrega as previsões de todas as árvores individuais. Para tarefas de classificação, a classe mais frequente entre as árvores é escolhida. Para tarefas de regressão, a média das previsões é calculada.

### Vantagens do Random Forest ✅

* Alta Precisão: Geralmente oferece melhor desempenho do que uma única árvore de decisão.
* Robustez: Menos propenso ao overfitting devido à agregação de múltiplas árvores.
* Capacidade de lidar com dados faltantes e variáveis categóricas.
* Importância das Features: Fornece medidas de importância das features, ajudando na interpretação do modelo.

### Desvantagens do Random Forest ❌

* Complexidade: Mais difícil de interpretar do que uma única árvore de decisão.
* Custo Computacional: Requer mais recursos computacionais para treinamento e previsão.
* Pode ser menos eficaz em conjuntos de dados muito pequenos.

### Métricas de Avaliação do Random Forest

As métricas de avaliação para o Random Forest incluem:

* Acurácia: Proporção de previsões corretas em relação ao total de previsões.
* Precisão: Proporção de verdadeiros positivos em relação ao total de positivos previstos.
* Revocação (Sensibilidade): Proporção de verdadeiros positivos em relação ao total de positivos reais.
* F1-Score: Média harmônica entre precisão e revocação.
* AUC-ROC: Área sob a curva ROC, que mede a capacidade de discriminação do modelo.
