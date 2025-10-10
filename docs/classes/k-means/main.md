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

