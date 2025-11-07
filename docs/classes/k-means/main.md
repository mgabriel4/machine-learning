# Projeto de Classificação de Casas com K-Means

Este projeto utiliza o algoritmo K-Means para agrupar casas com base em suas características e prever a faixa de preço (baixa, média, alta) no dataset Ames Housing. O objetivo é demonstrar como o K-Means pode ser aplicado em um problema de classificação não supervisionada.

## Exploração dos Dados (EDA)

O dataset Ames Housing contém diversas características das casas, como área, número de quartos, idade, entre outras. A análise exploratória dos dados (EDA) foi realizada para entender a distribuição das características e identificar possíveis correlações com o preço das casas.

Toda esta etapa foi realizada no arquivo [Exploração dos Dados (EDA)](/docs/classes/arvore-de-decisao/eda.py).

## Pré-processamento dos Dados

O dataset foi carregado e pré-processado para remover valores ausentes e normalizar as características. As principais etapas incluem:

1. Carregamento do dataset.
2. Tratamento de valores ausentes.
3. Normalização das características.
4. Seleção das características relevantes para o modelo.
5. Aplicação do K-Means para agrupar as casas.
6. Análise dos clusters formados e suas características médias.
7. Avaliação do modelo com base na distribuição dos clusters em relação às faixas de preço.
8. Visualização dos resultados.

As etapas 2, 3 e 4 foram realizadas no arquivo [Pré-processamento dos Dados](/docs/classes/k-means/processamento.ipynb).
Já a etapa de seleção das variáveis foi realizada no arquivo [Seleção de Variáveis](/docs/classes/knn/processamento.ipynb).

=== "Output"

    ```
        Valores ausentes por coluna:
        MS SubClass        0
        MS Zoning          0
        Lot Frontage       0
        Lot Area           0
        Street             0
        Lot Shape          0
        Land Contour       0
        Utilities          0
        Lot Config         0
        Land Slope         0
        Neighborhood       0
        Condition 1        0
        Condition 2        0
        Bldg Type          0
        House Style        0
        Overall Qual       0
        Overall Cond       0
        Year Built         0
        Year Remod/Add     0
        Roof Style         0
        Roof Matl          0
        Exterior 1st       0
        Exterior 2nd       0
        Mas Vnr Type       0
        Mas Vnr Area       0
        Exter Qual         0
        Exter Cond         0
        Foundation         0
        Bsmt Qual          0
        Bsmt Cond          0
        Bsmt Exposure      0
        BsmtFin Type 1     0
        BsmtFin SF 1       0
        BsmtFin Type 2     0
        BsmtFin SF 2       0
        Bsmt Unf SF        0
        Total Bsmt SF      0
        Heating            0
        Heating QC         0
        Central Air        0
        Electrical         0
        1st Flr SF         0
        2nd Flr SF         0
        Low Qual Fin SF    0
        Gr Liv Area        0
        Bsmt Full Bath     0
        Bsmt Half Bath     0
        Full Bath          0
        Half Bath          0
        Bedroom AbvGr      0
        Kitchen AbvGr      0
        Kitchen Qual       0
        TotRms AbvGrd      0
        Functional         0
        Fireplaces         0
        Garage Type        0
        Garage Yr Blt      0
        Garage Finish      0
        Garage Cars        0
        Garage Area        0
        Garage Qual        0
        Garage Cond        0
        Paved Drive        0
        Wood Deck SF       0
        Open Porch SF      0
        Enclosed Porch     0
        3Ssn Porch         0
        Screen Porch       0
        Pool Area          0
        Misc Val           0
        Mo Sold            0
        Yr Sold            0
        Sale Type          0
        Sale Condition     0
        SalePrice          0

        Total de linhas após remoção: 2930

                MS SubClass  Lot Frontage  Lot Area  Overall Qual  Overall Cond  \
        0           20         141.0     31770             6             5   
        1           20          80.0     11622             5             6   
        2           20          81.0     14267             6             6   
        3           20          93.0     11160             7             5   
        4           60          74.0     13830             5             5   
        5           60          78.0      9978             6             6   
        6          120          41.0      4920             8             5   
        7          120          43.0      5005             8             5   
        8          120          39.0      5389             8             5   
        9           60          60.0      7500             7             5   

        Year Built  Year Remod/Add  Mas Vnr Area  Exter Qual  Exter Cond  ...  \
        0        1960            1960         112.0           3           3  ...   
        1        1961            1961           0.0           3           3  ...   
        2        1958            1958         108.0           3           3  ...   
        3        1968            1968           0.0           4           3  ...   
        4        1997            1998           0.0           3           3  ...   
        5        1998            1998          20.0           3           3  ...   
        6        2001            2001           0.0           4           3  ...   
        7        1992            1992           0.0           4           3  ...   
        8        1995            1996           0.0           4           3  ...   
        9        1999            1999           0.0           3           3  ...   

        Sale Type_New  Sale Type_Oth  Sale Type_VWD  Sale Type_WD   \
        0          False          False          False           True   
        1          False          False          False           True   
        2          False          False          False           True   
        3          False          False          False           True   
        4          False          False          False           True   
        5          False          False          False           True   
        6          False          False          False           True   
        7          False          False          False           True   
        8          False          False          False           True   
        9          False          False          False           True   

        Sale Condition_Abnorml  Sale Condition_AdjLand  Sale Condition_Alloca  \
        0                   False                   False                  False   
        1                   False                   False                  False   
        2                   False                   False                  False   
        3                   False                   False                  False   
        4                   False                   False                  False   
        5                   False                   False                  False   
        6                   False                   False                  False   
        7                   False                   False                  False   
        8                   False                   False                  False   
        9                   False                   False                  False   

        Sale Condition_Family  Sale Condition_Normal  Sale Condition_Partial  
        0                  False                   True                   False  
        1                  False                   True                   False  
        2                  False                   True                   False  
        3                  False                   True                   False  
        4                  False                   True                   False  
        5                  False                   True                   False  
        6                  False                   True                   False  
        7                  False                   True                   False  
        8                  False                   True                   False  
        9                  False                   True                   False  

        [10 rows x 233 columns]

    ```

=== "Code"

    ```python

        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.cluster import KMeans

        df = pd.read_csv('/home/mgabriel4/Documentos/GitHub/machine-learning/data/AmesHousing.csv')

        df = df.drop(columns=['Order', 'PID'])
        df = df.drop(columns=['Pool QC', 'Misc Feature', 'Alley', 'Fence', 'Fireplace Qu'])

        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in num_cols:
            df[col].fillna(df[col].median(), inplace=True)

        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)

        nulos = df.isnull().sum()
        print("\nValores ausentes por coluna:")
        print(nulos.head(100).to_string())
        print(f"\nTotal de linhas após remoção: {len(df)}")

        variaveis_ordinal = {
            'Exter Qual': {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
            'Exter Cond': {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
            'Bsmt Qual': {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
            'Bsmt Cond': {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
            'Heating QC': {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
            'Kitchen Qual': {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
            'Garage Qual': {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
            'Garage Cond': {'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5},
            'Bsmt Exposure': {'No':1, 'Mn':2, 'Av':3, 'Gd':4},
            'BsmtFin Type 1': {'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6},
            'BsmtFin Type 2': {'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6},
            'Functional': {'Sal':1, 'Sev':2, 'Maj2':3, 'Maj1':4, 'Mod':5, 'Min2':6, 'Min1':7, 'Typ':8},
        }

        for col, mapping in variaveis_ordinal.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(int)

        variaveis_nominais = [
            'MS Zoning', 'Street', 'Lot Shape', 'Land Contour',
            'Utilities', 'Lot Config', 'Land Slope', 'Neighborhood', 'Condition 1',
            'Condition 2', 'Bldg Type', 'House Style', 'Roof Style', 'Roof Matl',
            'Exterior 1st', 'Exterior 2nd', 'Mas Vnr Type', 'Foundation',
            'Heating', 'Central Air', 'Electrical', 'Garage Type', 'Garage Finish',
            'Paved Drive', 'Sale Type', 'Sale Condition'
        ]

        df = pd.get_dummies(df, columns=variaveis_nominais)

        print(df.head(10))
        
    ```

---

## Divisão dos Dados

Os dados foram divididos em variáveis independentes (X) e a variável dependente (y), que representa a faixa de preço das casas. A variável dependente foi categorizada em três classes: baixa, média e alta.

A seleção das variáveis independentes mais importantes foi realizada no arquivo [Seleção de Variáveis](/docs/classes/knn/processamento.ipynb).

=== "Code"

    ```python

        df['Target'] = pd.qcut(df['SalePrice'], q=3, labels=['Baixa', 'Média', 'Alta'])

        df['Target'] = df['Target'].map({'Baixa':0, 'Média':1, 'Alta':2}).astype('int')

        x = df[['Overall Qual', 'Year Built', 'Exter Qual', 'Bsmt Qual', 'Gr Liv Area', 'Full Bath', 'Kitchen Qual', 'Garage Cars', 'Garage Area', 'Garage Finish_Unf']]
        y = df['Target'].values.astype('int')

    ```

---

## Treinamento do modelo

O modelo K-Means tem que ser treinado com o número de clusters (k) definido, pela técnica do Elbow Curve. Logo após o treinamento, os clusters formados foram analisados para entender suas características médias.

### Método Elbow Curve

Utilizamos este método para determinar o número ideal de clusters (k) para o K-Means. O gráfico do Elbow Curve mostra a soma dos quadrados dentro do cluster (inertia) em função do número de clusters. O ponto onde a redução da inertia começa a diminuir significativamente indica o valor ideal de k.

=== "Code"

    ```python

        inertia = []
        K = range(1, 10)

        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(x)
            inertia.append(kmeans.inertia_)

        plt.plot(K, inertia, 'bo-')
        plt.xlabel('Número de clusters (K)')
        plt.ylabel('Inércia (Soma das Distâncias)')
        plt.title('Método do Cotovelo')
        plt.savefig('/home/mgabriel4/Documentos/GitHub/machine-learning/docs/classes/k-means/img/elbow_method.png')
        plt.show()

    ```

![Elbow Curve](/docs/classes/k-means/img/elbow_method.png)

A partir do gráfico, escolhemos k=3 para o modelo K-Means.

---

### Aplicação do K-Means

O modelo K-Means foi treinado com k=3 clusters. Após o treinamento, os clusters foram analisados para entender suas características médias.

=== "Output"

    ```
        1    1040
        0     966
        2     924
        Name: Cluster, dtype: int64
    ```

=== "Code"

    ```python

        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Cluster'] = kmeans.fit_predict(x)

        print(df['Cluster'].value_counts())
        print(df.groupby('Cluster')[x.columns].mean())

        sns.scatterplot(
            data=df, x='Gr Liv Area', y='SalePrice',
            hue='Cluster', palette='viridis'
        )
        plt.title('Clusters - KMeans')
        plt.savefig('/home/mgabriel4/Documentos/GitHub/machine-learning/docs/classes/k-means/img/kmeans_clusters.png')
        plt.show()

    ```

![Clusters KMeans](/docs/classes/k-means/img/kmeans_clusters.png)

Interpretamos os clusters formados:

* **Cluster 0:** Casas com qualidade geral e área de estar menores, provavelmente correspondendo à faixa de preço baixa.
* **Cluster 1:** Casas com qualidade geral e área de estar intermediárias, provavelmente correspondendo à faixa de preço média.
* **Cluster 2:** Casas com qualidade geral e área de estar maiores, provavelmente correspondendo à faixa de preço alta.

---

## Avaliação do Modelo

A avaliação do modelo foi feita analisando a distribuição dos clusters em relação às faixas de preço (baixa, média, alta). Observamos que os clusters formados pelo K-Means correspondem razoavelmente bem às categorias de preço.

=== "Output"

    ```
        Cluster 0:
        Baixa: 0
        Média: 273
        Alta: 726

        Cluster 1:
        Baixa: 0
        Média: 0
        Alta: 243

        Cluster 2:
        Baixa: 981
        Média: 707
        Alta: 0
    ```

=== "Code"

    ```python

        for cluster in range(3):
            subset = df[df['Cluster'] == cluster]
            counts = subset['Target'].value_counts().sort_index()
            print(f"\nCluster {cluster}:")
            print(f"Baixa: {counts.get(0, 0)}")
            print(f"Média: {counts.get(1, 0)}")
            print(f"Alta: {counts.get(2, 0)}")

        df.groupby('Cluster')[x.columns].mean()

    ```

![Tabela de Clusters vs Valores médios dos features](/docs/classes/k-means/img/tabela_k-means.png)

Vemos que o modelo K-Means conseguiu agrupar as casas de forma que os clusters correspondem às faixas de preço, embora haja alguma sobreposição. Isso indica que o K-Means pode ser útil para segmentar o mercado imobiliário com base em características das casas.
