# Conceitos básicos

Estes são os conceitos da matéria de Machine Learning para que eu conseguisse elaborar os exercícios cobrados na matéria.

## Sobre IA

A Inteligência Artificial serve para automatizar a mão de obra humana. Ela pode ser categorizada em:

1. **Symbolic IA**: se concentra em representar o conhecimento da IA por meio de símbolos e regras. Usamos aqui o conhecimento do primeiro semestre da tabela verdade para saber se uma proposição é verdadeira ou falsa.

2. **Connectionist AI**: é a IA baseada em redes neurais e se concentra em cálculos e inferências matemáticas.

3. **Neuro-Symbolic AI**: é a intersecção entre o raciocínio simbólico com as redes neurais, o que potencializa os pontos fortes das outras IA's para que a Neuro-Symbolic AI consiga resolver problemas complexos e ao mesmo tempo aprender com os dados.

![Hierarquia da IA](../../assets/images/IA.png)

## Machine Learning

**Mas, o que é o Machine Learning?**

Esta técnica serve para que a máquina aprenda a partir dos dados, com o intuito de que o sistema consiga resolver problemas mais complexos e que melhorem seu desemepenho, sem que sejam programados necessariamente o tempo todo de execução.

Dentro do Machine Learning, as técnicas são divididas em duas grandes categorias principais: o aprendizado supervisionado e o aprendizado não supervisionado.

!!! info "Aprendizado supervisionado"

    O *aprendizado supervisionado* é quando treinamos um modelo com uma base de **dados rotulados**. Isso faz com que o sistema aprenda os padrões e faça previsões com uma base de dados novas e inéditas.

    USADO EM: tarefas de classificação e tarefas de regressão. -> abordagem eficaz quando existe uma relação entre as variáveis feature e target.

    * Os **dados rotulados** nada mais são do que o conjunto em que cada linha tem um valor de resposta correto, que você usa para ensinar o modelo. A variável target é considerada como o rótulo, por ser a variável resposta.

!!! info "Aprendizado não supervisionado"

    O *aprendizado não supervisionado* envolve uma base de dados não rotulada, ou seja, o próprio modelo deve encontrar padrões e relacionamentos dentro dos dados que não tem uma orientação explícita.

    USADO EM: análises exploratórias de dados e extração de recursos (clusterização e redução de número de recursos em um conjunto de dados) -> abordagem eficaz quando temos que descobrir estruturas ocultas em uma base de dados.

!!! info "Aprendizado por reforço"

    Essa técnica é mais comumente utilizada quando o sistema precisa tomar decisões sequenciais com o ambiente e receber um feedback na forma de ganho ou perca.

    USADO EM: jogos ou controle robótico.

As técnicas para aprendizado da máquina resolve uma variedade de problemas, e os principais deles são:

* Problema de classificação: envolve a previsão de categorias discretas (valores inteiros) ou rótulos futuros com base na feature.

* Problema de regressão: envolve a previsão de valores contínuos.

## Modelos

Temos modelos que são com aprendizado supervisionado e com aprendizado não supervisionado. Veja a distribuição dos algoritmos:

Para a construção dos modelos, normalmente seguimos algumas etapas. São elas:

![Etapas para a construção do modelo](img/etapas.png)

1. **Exploração dos Dados (EDA)**: Nesta etapa, os dados são analisados para entender suas características, identificar padrões e detectar possíveis problemas, como: valores ausentes ou outliers.

2. **Pré-processamento**: Os dados são limpos e preparados para a modelagem. Isso pode incluir a normalização, transformação de variáveis categóricas em numéricas (usando técnicas como One-Hot Encoding ou Label Encoding), tratamento dos valores ausentes. [Comparativo Técnico: Label Encoding vs One Hot Encoding](https://blog.dsacademy.com.br/comparativo-tecnico-label-encoding-vs-one-hot-encoding-em-machine-learning/#:~:text=Cada%20t%C3%A9cnica%20possui%20seu%20prop%C3%B3sito,que%20n%C3%A3o%20induz%20rela%C3%A7%C3%B5es%20inexistentes.)/

3. **Modelagem**: A árvore de decisão é construída a partir dos dados de treinamento. O algoritmo seleciona as features que melhor dividem os dados em cada nó, com base em critérios como entropia e ganho de informação.

4. **Resultados**: O modelo é avaliado usando o conjunto de teste para medir sua precisão e capacidade de generalização. Métricas como acurácia, precisão, recall e F1-score são comumente usadas.

5. **Melhorias**: Nesta etapa, é importante revisarmos os resultados do modelo e ver se tem como fazer melhorias, como tirar algumas variáveis insignificantes e etc.

6. **Conclusão**: Os resultados são interpretados e as decisões são tomadas com base nas previsões do modelo. A árvore de decisão pode ser visualizada para entender como as decisões foram feitas.

---

## Referências

* [Machine Learning Mastery - Decision Trees](https://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/)
* [Machine Learning Mastery - KNN](https://machinelearningmastery.com/tutorial-to-k-nearest-neighbors-in-python/)
* [Machine Learning Mastery - K-Means](https://machinelearningmastery.com/k-means-clustering-for-machine-learning/)
* [Scikit-Learn - Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
* [Scikit-Learn - KNN](https://scikit-learn.org/stable/modules/neighbors.html)
* [Scikit-Learn - K-Means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
* [Wikipedia - Decision Trees](https://en.wikipedia.org/wiki/Decision_tree)
* [Wikipedia - KNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
* [Wikipedia - K-Means](https://en.wikipedia.org/wiki/K-means_clustering)
* [Towards Data Science - Decision Trees](https://towardsdatascience.com/decision-trees-in-machine-learning-641b9c4e8052)
* [Towards Data Science - KNN](https://towardsdatascience.com/k-nearest-neighbors-knn-algorithm-bd72f3f4f3e7)
* [Towards Data Science - K-Means](https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a)
* [GeeksforGeeks - Decision Trees](https://www.geeksforgeeks.org/decision-tree-in-machine-learning/)
* [GeeksforGeeks - KNN](https://www.geeksforgeeks.org/k-nearest-neighbors-knn-algorithm-in-python/)
* [GeeksforGeeks - K-Means](https://www.geeksforgeeks.org/k-means-clustering-algorithm-in-python/)
