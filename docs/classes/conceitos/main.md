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

    O *aprendizado supervisionado* é quando treinamos um modelo com uma base de dados rotulados. Isso faz com que o sistema aprenda os padrões e faça previsões com uma base de dados novas e inéditas.

    USADO EM: tarefas de classificação e tarefas de regressão. -> abordagem eficaz quando existe uma relação entre as variáveis feature e target.

!!! info "Aprendizado não supervisionado"

    O *aprendizado não supervisionado* envolvde uma base de dados não rotulada, ou seja, o próprio modelo deve encontrar padrões e relacionamentos dentro dos dados que não tem uma orientação explícita.

    USADO EM: análises exploratórias de dados e extração de recursos (clusterização e redução de número de recursos em um conjunto de dados) -> abordagem eficaz quando temos que descobrir estruturas ocultas em uma base de dados.

!!! info "Aprendizado por reforço"

    Essa técnica é mais comumente utilizada quando o sistema precisa tomar decisões sequenciais com o ambiente e receber um feedback na forma de ganho ou perca.

    USADO EM: jogos ou controle robótico.

As técnicas para aprendizado da máquina resolve uma variedade de problemas, e os principais deles são:

* Problema de classificação: envolve a previsão de categorias discretas (valores inteiros) ou rótulos futuros com base na feature.

* Problema de regressão: envolve a previsão de valores contínuos.

## Árvore de decisão

As árvores de decisão são uma técnica popular de aprendizado de máquina supervisionado usada para classificação e regressão. Elas representam decisões e suas possíveis consequências em uma estrutura hierárquica, facilitando a interpretação dos resultados.

### Etapas da árvore

``` mermaid

    flowchart TD
        A[Exploração dos Dados] --> B[Pré-processamento]
        B --> C[Modelagem]
        C --> D[Resultados]
        D --> E[Conclusão]

```

## KNN (K-Nearest Neighbors)

O KNN é um algoritmo de aprendizado supervisionado usado para problemas de classificação e regressão. Ele é baseado na ideia de que objetos semelhantes estão próximos uns dos outros no espaço de características. É válido ressaltar que o KNN é bom para conjuntos de dados pequenos e médios, e é simples de entender e implementar.

### Funcionamento

1. **Escolha do K**: O primeiro passo é escolher o número de vizinhos (K) que serão considerados para a classificação ou regressão. Este passo é crucial, pois um valor muito pequeno pode tornar o modelo sensível ao ruído, enquanto um valor muito grande pode suavizar demais as fronteiras de decisão. Para que não haja uma escolha do K equivocada, faz-se a técnica de *validação cruzada*.

2. **Cálculo da Distância**: Para classificar um novo ponto, o algoritmo calcula a distância entre esse ponto e todos os pontos do conjunto de treinamento. As métricas de distâncias mais comuns são a Euclidiana, Manhattan e Minkowski.

3. **Identificação dos Vizinhos**: O algoritmo seleciona os K pontos mais próximos do conjunto de treinamento.

4. **Classificação ou Regressão**:
   * **Classificação**: O rótulo do novo ponto é determinado pelos rótulos frequentes dos K vizinhos. (Ou seja, a moda dos rótulos)
   * **Regressão**: O valor do novo ponto é determinado pela média (ou mediana) dos valores dos K vizinhos.

### Vantagens

* Simplicidade: O KNN é fácil de entender e implementar.
* Flexibilidade: Pode ser usado para classificação e regressão.
* Não paramétrico: Não faz suposições sobre a distribuição dos dados.

### Desvantagens

* Custo computacional: O KNN pode ser lento, especialmente com grandes conjuntos de dados, pois precisa calcular a distância de todos os pontos.
* Sensibilidade a ruídos: O algoritmo pode ser afetado por outliers e ruídos nos dados.
* Escolha do K: A escolha do valor de K pode impactar significativamente o desempenho do modelo.
