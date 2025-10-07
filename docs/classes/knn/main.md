# Projeto: Classificação das casas dos EUA com KNN

## 1. Exploração dos Dados (EDA)

Nesta etapa, foi realizada a análise exploratória do dataset [AmesHousing.csv](https://www.kaggle.com/datasets/ritwikb3/heart-disease-cleveland?resource=download), que contém informações sobre as casas em Ames, Iowa.

Toda esta parte foi analisada anteriormente no arquivo [processamento.py](/docs/classes/arvore-de-decisao/eda.py).

=== "Output"

    ```
    Primeiras 5 linhas do dataset:

    age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  ca  thal  target
    0   63    1   0       145   233    1        2      150      0      2.3      2   0     2       0
    1   67    1   3       160   286    0        2      108      1      1.5      1   3     1       1
    2   67    1   3       120   229    0        2      129      1      2.6      1   2     3       1
    3   37    1   2       130   250    0        0      187      0      3.5      2   0     1       0
    4   41    0   1       130   204    0        2      172      0      1.4      0   0     1       0

    Informações do dataset:
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 303 entries, 0 to 302
    Data columns (total 14 columns):
    #   Column    Non-Null Count  Dtype
    ---  ------    --------------  -----
    0   age       303 non-null    int64
    1   sex       303 non-null    int64
    2   cp        303 non-null    int64
    3   trestbps  303 non-null    int64
    4   chol      303 non-null    int64
    5   fbs       303 non-null    int64
    6   restecg   303 non-null    int64
    7   thalach   303 non-null    int64
    8   exang     303 non-null    int64
    9   oldpeak   303 non-null    float64
    10  slope     303 non-null    int64
    11  ca        303 non-null    int64
    12  thal      303 non-null    int64
    13  target    303 non-null    int64
    dtypes: float64(1), int64(13)
    memory usage: 33.3 KB
    None

    Estatísticas descritivas:
    
    age         sex          cp    trestbps  ...       slope          ca        thal      targetcount  303.000000  303.000000  303.000000  303.000000  ...  303.000000  303.000000  303.000000  303.000000mean    54.438944    0.679868    2.158416  131.689769  ...    0.600660    0.663366    1.831683    0.458746std      9.038662    0.467299    0.960126   17.599748  ...    0.616226    0.934375    0.956705    0.499120min     29.000000    0.000000    0.000000   94.000000  ...    0.000000    0.000000    1.000000    0.00000025%     48.000000    0.000000    2.000000  120.000000  ...    0.000000    0.000000    1.000000    0.00000050%     56.000000    1.000000    2.000000  130.000000  ...    1.000000    0.000000    1.000000    0.00000075%     61.000000    1.000000    3.000000  140.000000  ...    1.000000    1.000000    3.000000    1.000000max     77.000000    1.000000    3.000000  200.000000  ...    2.000000    3.000000    3.000000    1.000000
    [8 rows x 14 columns]

    Valores ausentes por coluna:
    
    age         0
    sex         0
    cp          0
    trestbps    0
    chol        0
    fbs         0
    restecg     0
    thalach     0
    exang       0
    oldpeak     0
    slope       0
    ca          0
    thal        0
    target      0
    dtype: int64
    ```

=== "Code"

    ```python
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    dados = pd.read_csv('data/heart.csv')

    print("Primeiras 5 linhas do dataset:")
    print(dados.head())
    print("\nInformações do dataset:")
    print(dados.info())
    print("\nEstatísticas descritivas:")
    print(dados.describe(include='all'))
    print("\nValores ausentes por coluna:")
    print(dados.isnull().sum())
    ```

---

## 2. Pré-processamento

Feito anteriormente no arquivo [processamento.py](../arvore-de-decisao/processamento.py), o pré-processamento incluiu:

* Tratamento de valores ausentes, como preenchimento por média/mediana/moda e remoção de linhas/colunas.
* Codificação de variáveis categóricas, utilizando as técnicas de One-Hot Encoding e Label Encoding.

Para o modelo KNN, selecionei as 10 features mais relevantes utilizando o método `SelectKBest` com a função de pontuação `f_classif`. **Mas por quê eu tinha que fazer essa seleção?** Porque o KNN é um algoritmo baseado em distância, e muitas features irrelevantes ou redundantes podem introduzir ruído e prejudicar o desempenho do modelo. Ao selecionar as features mais importantes, podemos melhorar a precisão e a eficiência do KNN.

É importante também normalizar ou padronizar as features numéricas para evitar que variáveis com escalas maiores dominem a distância calculada entre os pontos. E aqui eu utilizei o `StandardScaler` para padronizar as features selecionadas. É aconselhável fazer isso **após** a seleção das features, para garantir que a padronização seja aplicada apenas às variáveis relevantes.

A padronização é aplicada apenas nas features, pois a varíavel preditora (target) é categórica e não deve ser alterada.

=== "Output"

    ```
    As 10 variáveis mais relevantes para o modelo KNN:
    ['Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Total Bsmt SF', '1st Flr SF', 'Full Bath', 'Year Built', 'Year Remod/Add', 'TotRms AbvGrd', 'Garage Area']
    ```

=== "Code"

    ```python
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.preprocessing import StandardScaler

    selector = SelectKBest(score_func=f_classif, k=10)
    x_new = selector.fit_transform(x, y)
    selected_features = df.drop(columns=['SalePrice', 'Target']).columns[selector.get_support()]

    print("\nAs 10 variáveis mais relevantes para o modelo KNN:")
    print(selected_features.tolist())

    x_scaled = StandardScaler().fit_transform(x_new)
    
    ```

---

## 3. Divisão dos Dados

Com o intuito de ver se minha hipótese é verdadeira ou não, eu treinei o meu modelo dividindo os dados em conjuntos de treino e teste (70% por 30%).

=== "Code"

    ```python
    # Seleção das features e target
    x = df[['Overall Qual', 'Year Built', 'Exter Qual', 'Bsmt Qual', 'Gr Liv Area', 'Kitchen Qual', 'Garage Cars', 'Garage Area', 'Overall Qual_scaled', 'Garage Finish_Unf']].values
    y = df['Target'].values.astype('int')

    # Divisão dos dados
    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)
    ```

=== "Explicação"

    * As variáveis de entrada (features -> variáveis x) e saída (target -> variável y) foram definidas.

    * Split 70/30 para treino e teste, garantindo avaliação justa do modelo.

---

## 4. Treinamento do Modelo

Antes do treinamento, utilizei a validação cruzada para encontrar o valor ideal de k (número de vizinhos) para o KNN. Após testar valores de k de 1 a 20, o melhor valor encontrado foi k=11, com uma acurácia média de aproximadamente 0.79.

=== "Output"

    ```
    Melhor k: 11
    Acurácia média com esse k: 0.791
    ```

=== "Code"

    ```python
    #testar o k ideal com validação cruzada
    from sklearn.model_selection import GridSearchCV
    param_grid = {'n_neighbors': range(1, 21)}  # testa k de 1 a 20

    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid.fit(x_scaled, y)

    print(f"Melhor k: {grid.best_params_['n_neighbors']}")
    print(f"Acurácia média com esse k: {grid.best_score_:.3f}")
    ```

=== "Explicação"

        * Utilizou-se o `GridSearchCV` para encontrar o melhor valor de k, testando valores de 1 a 20 com validação cruzada de 5 folds. Para cada valor de k, a acurácia média foi calculada.

        * O modelo foi ajustado aos dados de treino.

---

Para a confirmação do valor de k, construí um gráfico de acurácia média versus valores de k. O gráfico mostra que a acurácia atinge um pico em k=11, confirmando a escolha do melhor valor de k.

=== "Gráfico"
    ![Gráfico de Acurácia vs k](../../knn/img/acuracia.png)

=== "Output"

    ```
    Melhor valor de k: 11 (Acurácia média = 0.722)
    ```

=== "Code"

    ```python
    from sklearn.model_selection import cross_val_score
    import matplotlib.pyplot as plt
    import numpy as np

    k_values = range(1, 21)
    mean_scores = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, x_scaled, y, cv=5, scoring='accuracy')
        mean_scores.append(scores.mean())

    plt.figure(figsize=(8, 4))
    plt.plot(k_values, mean_scores, marker='o', linestyle='-')
    plt.title("Acurácia média (5-fold) vs Número de vizinhos (k)")
    plt.xlabel("Número de vizinhos (k)")
    plt.ylabel("Acurácia média")
    plt.grid(True)
    plt.savefig('/home/mgabriel4/Documentos/GitHub/machine-learning/docs/classes/knn/img/acuracia.png')
    plt.show()

    best_k = k_values[np.argmax(mean_scores)]
    print(f"Melhor valor de k: {best_k} (Acurácia média = {max(mean_scores):.3f})")
    ```

=== "Explicação"

    * O gráfico ilustra a relação entre o número de vizinhos (k) e a acurácia média obtida por validação cruzada.

    * O pico em k=11 reforça a escolha do melhor valor de k para o modelo.

Agora, após a comparação entre os resultados das duas técnicas para descobrir o valor k mais adequado, com o valor de k definido, treinei o modelo KNN com k=11 usando os dados de treino.

=== "Output"

    ```
    Accuracy: 0.80
    ```

=== "Code"

    ```python
    knn = KNeighborsClassifier(n_neighbors=11)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    ```

---

## 5. Avaliação do Modelo

O desempenho do modelo foi avaliado com métricas de classificação e visualização da matriz de confusão.

=== "Output"

    ```
    Acurácia: 0.8043230944254836

    Relatório de classificação:
                precision    recall  f1-score   support

            0       0.80      0.81      0.81       286
            1       0.68      0.70      0.69       274
            2       0.92      0.88      0.90       319

        accuracy                           0.80       879
    macro avg       0.80      0.80      0.80       879
    weighted avg       0.81      0.80      0.81       879
    ```

=== "Code"

    ```python
    print("Acurácia:", accuracy_score(y_test, predictions))
    print("\nRelatório de classificação:\n", classification_report(y_test, predictions))

    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.show()
    ```

=== "Gráfico"
    ![Matriz de Confusão](../../knn/img/matriz_confusao.png)

=== "Explicação"

    * O modelo foi avaliado por métricas como precisão, recall e F1-score.

    * A matriz de confusão foi visualizada para interpretação dos resultados.

---

A matriz de confusão indica que o modelo KNN apresentou boa capacidade de discriminar as classes de preço. As casas de categoria “Baixa” e “Alta” foram corretamente classificadas na maior parte dos casos (acima de 80% de acerto), enquanto a classe intermediária (“Média”) apresentou maior número de erros, sendo frequentemente confundida com as categorias vizinhas. Esse comportamento é esperado, pois os imóveis de preço médio compartilham características tanto de casas baratas quanto de casas de alto padrão. No geral, a acurácia global do modelo foi de aproximadamente 79%, indicando um desempenho satisfatório na classificação do preço dos imóveis.

### 5.1 Gráfico do Limite de Decisão

O gráfico abaixo representa o limite de decisão do modelo KNN treinado com k=11. As regiões coloridas indicam a classificação prevista pelo modelo para diferentes combinações das duas features selecionadas: 'Overall Qual' (qualidade geral) e 'Gr Liv Area' (área de estar acima do solo). Os pontos pretos representam os dados de teste, onde cada ponto é uma casa com sua respectiva classificação real.

![Gráfico do Limite de Decisão](../../knn/img/fronteira_decisao.png)

=== "Code"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    # Seleciona duas features para visualização
    feature1 = 0  # Índice da primeira feature (Overall Qual)
    feature2 = 1  # Índice da segunda feature (Gr Liv Area)

    x_min, x_max = x_scaled[:, feature1].min() - 1, x_scaled[:, feature1].max() + 1
    y_min, y_max = x_scaled[:, feature2].min() - 1, x_scaled[:, feature2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.scatter(X_test[:, feature1], X_test[:, feature2], c=y_test, edgecolor='k', marker='o', s=100, cmap=plt.cm.RdYlBu)
    plt.xlabel('Overall Qual (padronizado)')
    plt.ylabel('Gr Liv Area (padronizado)')
    plt.title('Limite de Decisão do KNN (k=11)')
    plt.show()
    ```

=== "Explicação"

    * O gráfico ilustra como o modelo KNN classifica diferentes regiões do espaço de features.

    * As áreas coloridas representam as previsões do modelo, enquanto os pontos pretos indicam os dados de teste reais.

    * Podemos utilizar apenas duas features para visualizar o limite de decisão, mas o modelo KNN foi treinado com todas as 10 features selecionadas.
---

## 6. Relatório Final

O modelo KNN se mostrou eficiente para a tarefa de classificação das casas do dataset Ames Housing, obtendo uma acurácia global de aproximadamente 80%. Após a etapa de pré-processamento — que incluiu o tratamento de valores ausentes, codificação adequada das variáveis categóricas e padronização das features —, foi realizada a seleção das 10 variáveis mais relevantes utilizando o método SelectKBest, o que contribuiu para reduzir ruído e melhorar o desempenho do modelo.

Por meio da validação cruzada com GridSearchCV, identificou-se que o melhor número de vizinhos foi k = 11, ponto em que a acurácia média atingiu o seu valor máximo. A matriz de confusão indicou que o modelo apresentou excelente desempenho nas classes “Baixa” e “Alta”, com taxas de acerto acima de 80%, e um desempenho moderado na classe “Média”, que tende naturalmente a se confundir com as faixas adjacentes de preço.

A análise gráfica do limite de decisão demonstrou que o KNN foi capaz de delimitar regiões claras de decisão no espaço de atributos, especialmente entre as classes extremas, refletindo a coerência dos padrões de qualidade e metragem presentes no conjunto de dados.

Em suma, o modelo apresentou bom equilíbrio entre simplicidade e desempenho, sendo capaz de generalizar adequadamente os padrões do conjunto de dados. Entretanto, possíveis melhorias poderiam envolver o uso de ponderação de distância (weights='distance'), métodos de redução de dimensionalidade (PCA) ou até mesmo o ajuste de parâmetros adicionais para explorar ainda mais o potencial do KNN no contexto do mercado imobiliário.
