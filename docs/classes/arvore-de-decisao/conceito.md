# Árvore de decisão

As árvores de decisão são uma técnica popular de aprendizado de máquina supervisionado usada para classificação e regressão. Elas representam decisões e suas possíveis consequências em uma estrutura hierárquica, facilitando a interpretação dos resultados.

## Funcionamento

1. **Divisão Recursiva**: A árvore de decisão começa com um nó raiz que representa todo o conjunto de dados. O algoritmo então divide os dados em subconjuntos com base em uma feature que maximiza a separação entre as classes (para classificação) ou minimiza o erro (para regressão). Esse processo é repetido recursivamente para cada subconjunto, criando nós filhos até que um critério de parada seja atingido (como profundidade máxima da árvore ou número mínimo de amostras em um nó).

2. **Nós e Folhas**: Cada nó interno representa uma decisão baseada em uma feature, enquanto as folhas representam as previsões finais (rótulos de classe ou valores contínuos).

### Vantagens da Árvore de Decisão ✅

* Fácil de interpretar e visualizar.
* Pode lidar com dados categóricos e numéricos.
* Requer pouca preparação de dados.
* Pode capturar relações não lineares.

### Desvantagens da Árvore de Decisão ❌

* Propenso ao overfitting, especialmente com árvores profundas.
* Sensível a pequenas variações nos dados.
* Pode ser instável, pois pequenas mudanças nos dados podem levar a árvores muito diferentes.

### Métricas de Avaliação da Árvore de Decisão

**Entropia**: A entropia é uma medida da incerteza ou impureza em um conjunto de dados. Em termos simples, ela quantifica o grau de desordem ou aleatoriedade em um sistema. Em aprendizado de máquina, a entropia é usada para avaliar a qualidade das divisões em uma árvore de decisão.

**Ganho de Informação**: O ganho de informação é uma métrica que quantifica a redução da entropia após uma divisão dos dados com base em uma feature específica. Em outras palavras, ele mede o quanto a incerteza sobre a variável alvo diminui quando os dados são divididos com base em uma determinada feature. O ganho de informação é calculado como a diferença entre a entropia do conjunto de dados original e a entropia ponderada dos subconjuntos resultantes da divisão.

**Índice Gini**: O índice Gini é uma medida de impureza ou pureza usada em árvores de decisão para avaliar a qualidade das divisões dos dados. Ele quantifica a probabilidade de um elemento ser classificado incorretamente se fosse rotulado aleatoriamente de acordo com a distribuição das classes no conjunto de dados. O índice Gini varia entre 0 (pureza máxima, onde todos os elementos pertencem à mesma classe) e 0,5 (impureza máxima, onde as classes estão igualmente distribuídas).
