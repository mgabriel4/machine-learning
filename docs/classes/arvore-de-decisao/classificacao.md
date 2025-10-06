# Projeto: Baixa, média e alta renda, qual casa você pode comprar nos EUA?

## Objetivo

O objetivo deste projeto é utilizar um modelo de árvore de decisão para classificar imóveis em diferentes faixas de preço (baixa, média e alta renda) com base em suas características.

## 1. Exploração dos Dados (EDA)

Nesta etapa, foi realizada a análise exploratória do dataset [AmesHousing.csv](https://www.kaggle.com/datasets/hsumedh1507/ames-housing-dataset), verificando as primeiras linhas, informações gerais, estatísticas descritivas, valores ausentes e visualização de algumas variáveis categóricas.

=== "Output"

    ```
    Primeiras 5 linhas do dataset:
        Order        PID  MS SubClass MS Zoning  Lot Frontage  Lot Area  ... Misc Val Mo Sold Yr Sold Sale Type Sale Condition SalePrice
    0      1  526301100           20        RL         141.0     31770  ...        0       5    2010       WD          Normal    215000
    1      2  526350040           20        RH          80.0     11622  ...        0       6    2010       WD          Normal    105000
    2      3  526351010           20        RL          81.0     14267  ...    12500       6    2010       WD          Normal    172000
    3      4  526353030           20        RL          93.0     11160  ...        0       4    2010       WD          Normal    244000
    4      5  527105010           60        RL          74.0     13830  ...        0       3    2010       WD          Normal    189900

    [5 rows x 82 columns]

    Informações do dataset:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2930 entries, 0 to 2929
    Data columns (total 82 columns):
    #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
    0   Order            2930 non-null   int64  
    1   PID              2930 non-null   int64  
    2   MS SubClass      2930 non-null   int64  
    3   MS Zoning        2930 non-null   object 
    4   Lot Frontage     2440 non-null   float64
    5   Lot Area         2930 non-null   int64  
    6   Street           2930 non-null   object 
    7   Alley            198 non-null    object 
    8   Lot Shape        2930 non-null   object 
    9   Land Contour     2930 non-null   object 
    10  Utilities        2930 non-null   object 
    11  Lot Config       2930 non-null   object 
    12  Land Slope       2930 non-null   object 
    13  Neighborhood     2930 non-null   object 
    14  Condition 1      2930 non-null   object 
    15  Condition 2      2930 non-null   object 
    16  Bldg Type        2930 non-null   object 
    17  House Style      2930 non-null   object 
    18  Overall Qual     2930 non-null   int64  
    19  Overall Cond     2930 non-null   int64  
    20  Year Built       2930 non-null   int64  
    21  Year Remod/Add   2930 non-null   int64  
    22  Roof Style       2930 non-null   object 
    23  Roof Matl        2930 non-null   object 
    24  Exterior 1st     2930 non-null   object 
    25  Exterior 2nd     2930 non-null   object 
    26  Mas Vnr Type     1155 non-null   object 
    27  Mas Vnr Area     2907 non-null   float64
    28  Exter Qual       2930 non-null   object 
    29  Exter Cond       2930 non-null   object 
    30  Foundation       2930 non-null   object 
    31  Bsmt Qual        2850 non-null   object 
    32  Bsmt Cond        2850 non-null   object 
    33  Bsmt Exposure    2847 non-null   object 
    34  BsmtFin Type 1   2850 non-null   object 
    35  BsmtFin SF 1     2929 non-null   float64
    36  BsmtFin Type 2   2849 non-null   object 
    37  BsmtFin SF 2     2929 non-null   float64
    38  Bsmt Unf SF      2929 non-null   float64
    39  Total Bsmt SF    2929 non-null   float64
    40  Heating          2930 non-null   object 
    41  Heating QC       2930 non-null   object 
    42  Central Air      2930 non-null   object 
    43  Electrical       2929 non-null   object 
    44  1st Flr SF       2930 non-null   int64  
    45  2nd Flr SF       2930 non-null   int64  
    46  Low Qual Fin SF  2930 non-null   int64  
    47  Gr Liv Area      2930 non-null   int64  
    48  Bsmt Full Bath   2928 non-null   float64
    49  Bsmt Half Bath   2928 non-null   float64
    50  Full Bath        2930 non-null   int64  
    51  Half Bath        2930 non-null   int64  
    52  Bedroom AbvGr    2930 non-null   int64  
    53  Kitchen AbvGr    2930 non-null   int64  
    54  Kitchen Qual     2930 non-null   object 
    55  TotRms AbvGrd    2930 non-null   int64  
    56  Functional       2930 non-null   object 
    57  Fireplaces       2930 non-null   int64  
    58  Fireplace Qu     1508 non-null   object 
    59  Garage Type      2773 non-null   object 
    60  Garage Yr Blt    2771 non-null   float64
    61  Garage Finish    2771 non-null   object 
    62  Garage Cars      2929 non-null   float64
    63  Garage Area      2929 non-null   float64
    64  Garage Qual      2771 non-null   object 
    65  Garage Cond      2771 non-null   object 
    66  Paved Drive      2930 non-null   object 
    67  Wood Deck SF     2930 non-null   int64  
    68  Open Porch SF    2930 non-null   int64  
    69  Enclosed Porch   2930 non-null   int64  
    70  3Ssn Porch       2930 non-null   int64  
    71  Screen Porch     2930 non-null   int64  
    72  Pool Area        2930 non-null   int64  
    73  Pool QC          13 non-null     object 
    74  Fence            572 non-null    object 
    75  Misc Feature     106 non-null    object 
    76  Misc Val         2930 non-null   int64  
    77  Mo Sold          2930 non-null   int64  
    78  Yr Sold          2930 non-null   int64  
    79  Sale Type        2930 non-null   object 
    80  Sale Condition   2930 non-null   object 
    81  SalePrice        2930 non-null   int64  
    dtypes: float64(11), int64(28), object(43)
    memory usage: 1.8+ MB
    None

    Estatísticas descritivas:
                Order           PID  MS SubClass MS Zoning  Lot Frontage  ...      Mo Sold      Yr Sold Sale Type Sale Condition      SalePrice
    count   2930.00000  2.930000e+03  2930.000000      2930   2440.000000  ...  2930.000000  2930.000000      2930           2930    2930.000000
    unique         NaN           NaN          NaN         7           NaN  ...          NaN          NaN        10              6            NaN
    top            NaN           NaN          NaN        RL           NaN  ...          NaN          NaN       WD          Normal            NaN
    freq           NaN           NaN          NaN      2273           NaN  ...          NaN          NaN      2536           2413            NaN
    mean    1465.50000  7.144645e+08    57.387372       NaN     69.224590  ...     6.216041  2007.790444       NaN            NaN  180796.060068
    std      845.96247  1.887308e+08    42.638025       NaN     23.365335  ...     2.714492     1.316613       NaN            NaN   79886.692357
    min        1.00000  5.263011e+08    20.000000       NaN     21.000000  ...     1.000000  2006.000000       NaN            NaN   12789.000000
    25%      733.25000  5.284770e+08    20.000000       NaN     58.000000  ...     4.000000  2007.000000       NaN            NaN  129500.000000
    50%     1465.50000  5.354536e+08    50.000000       NaN     68.000000  ...     6.000000  2008.000000       NaN            NaN  160000.000000
    75%     2197.75000  9.071811e+08    70.000000       NaN     80.000000  ...     8.000000  2009.000000       NaN            NaN  213500.000000
    max     2930.00000  1.007100e+09   190.000000       NaN    313.000000  ...    12.000000  2010.000000       NaN            NaN  755000.000000

    [11 rows x 82 columns]

    Valores ausentes por coluna:
    Order               0
    PID                 0
    MS SubClass         0
    MS Zoning           0
    Lot Frontage      490
                    ... 
    Mo Sold             0
    Yr Sold             0
    Sale Type           0
    Sale Condition      0
    SalePrice           0
    Length: 82, dtype: int64
    ```

=== "Code"

    ```python

    ```

=== "Explicação"

    Utilizamos a biblioteca Pandas para a análise de dados e a biblioteca Matplotlib.pyplot para fazer os gráficos para uma melhor visualização dos dados.
 
    Logo após carregarmos a base de dados:
    
    * base = pd.read_csv('caminho/para/a/base')

    Utilizamos comandos para a análise exploratória da base:

    * base.head() -> sem a especificação de quantas linhas puxar, o head imprime as primeiras 5 linhas da sua base de dados.

    * base.info() -> mostra as colunas da base de dados e seus respectivos tipos.

    * base.describe(include='all') -> imprime as estatísticas descritivas das colunas da base de dados, como a frequência, a média, o desvio padrão e etc.

    * base.isnull().sum() -> soma todos os valores nulos por coluna.

    Após finalizar esse processo com a análise exploratória, precisamos fazer a visualização da análise realizada. Por isso, utilizamos os seguintes comandos:
    
    * plt.style.use('ggplot') -> sendo plt a biblioteca do matplotlib, esse comando serve para escolhermos o estilo que 
    
    * plt.figure(figsize=(15, 10)) -> cria uma nova figura para colocar os gráficos.

    * plt.subplot(2, 2, 1) -> este comando divide a figura em uma grade de 2 linhas por 2 colunas. O último parâmetro é a posição do gráfico.
    
    * base['nome_coluna'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(6,6)) -> constrói um gráfico de pizza com a coluna escolhida da base. O comando autopct exibe as porcentagens em cada fatia.
    
    * plt.title('Título') -> intitula o gráfico.
    
    * plt.ylabel('') -> remove o label do eixo Y, comando utilizado apenas em gráficos de pizza

    * base['nome_coluna'].value_counts().head(5).plot(kind='bar') -> constrói um gráfico de barra, e com o comando head(5) irá aparecer apenas os 5 mais frequentes daquela coluna.
  
    * plt.xticks(rotation=45) -> rotaciona o gráfico de barra a 45° para uma melhor visualização.

    * plt.savefig('caminho/que/deseja/salvar') -> salva a figura completa que foi iniciada lá no início do código no caminho que foi apontado.

    * plt.show() -> exibe a figura na tela.

### Gráficos gerados na análise exploratória

![Valores Ausentes](../../arvore-de-decisao/img/valores_ausentes.png)

A partir deste gráfico, podemos observar que algumas variáveis possuem uma quantidade significativa de valores ausentes. Isso pode impactar a análise e o desempenho do modelo, sendo necessário considerar estratégias para lidar com esses dados faltantes, como imputação ou remoção dessas variáveis.

![Variáveis Categóricas](../../arvore-de-decisao/img/categoricas.png)

No gráfico acima, podemos observar a distribuição das variáveis categóricas selecionadas. Isso nos ajuda a entender a frequência de diferentes categorias dentro dessas variáveis, o que pode ser útil para identificar padrões ou tendências no conjunto de dados. Um exemplo disso é o gráfico de pizza que mostra a predominância de um tipo específico de rua, indicando que a maioria das propriedades está localizada em ruas pavimentadas. E o gráfico de classe de zona residencial, onde a maioria das propriedades está situada em zonas residenciais (RL: Residential Low Density).

![Distribuição de Variáveis Numéricas](../../arvore-de-decisao/img/dist_geral_numericas.png)

No gráfico acima, podemos observar a distribuição das variáveis numéricas selecionadas. Isso nos ajuda a entender a frequência de diferentes valores dentro dessas variáveis, o que pode ser útil para identificar padrões ou tendências no conjunto de dados. Um exemplo disso é a distribuição do preço de venda, que apresenta uma assimetria à direita, indicando que a maioria das propriedades tem preços mais baixos, com algumas propriedades de alto valor.

![Boxplots](../../arvore-de-decisao/img/boxplots.png)

Os boxplots acima ilustram a relação entre a qualidade geral das casas e o preço de venda. Podemos observar que, em geral, casas com melhor qualidade tendem a ter preços de venda mais altos. No entanto, também há uma variação significativa nos preços dentro de cada categoria de qualidade, indicando que outros fatores além da qualidade geral também influenciam o preço de venda. Além disso, temos um boxplot sobre o preço de venda e o bairro onde a casa está localizada. Podemos observar que certos bairros, como NoRidge e NridgHt, apresentam preços de venda significativamente mais altos em comparação com outros bairros. Isso sugere que a localização é um fator importante na determinação do valor das propriedades.

![Matriz de Correlação](../../arvore-de-decisao/img/matriz_correlacao.png)

A matriz de correlação acima mostra a relação entre várias variáveis numéricas no conjunto de dados. Podemos observar que algumas variáveis, como "Gr Liv Area" (área de vida acima do solo) e "Overall Qual" (qualidade geral), têm uma correlação positiva forte com o preço de venda ("SalePrice"). Isso indica que casas com maior área de vida e melhor qualidade tendem a ter preços de venda mais altos. Outras variáveis, como "Lot Area" (área do lote) e "Garage Area" (área da garagem), também mostram correlações positivas, mas em menor grau. Essas informações podem ser úteis para identificar quais características das casas são mais influentes na determinação do preço de venda.

---

## 2. Pré-processamento

Como primeira fase do pré-processamento, precisamos tratar os valores ausentes. Como visto anteriormente, principalmente no gráfico gerado, temos muitos valores nulos a serem tratados. Abaixo estão listados a quantidade dos valores nulos em todas as variáveis:

* Lot Frontage (490 valores nulos),
* Mas Vnr Type (1775 valores nulos),
* Mas Vnr Area (23 valores nulos),
* Bsmt Qual (80 valores nulos),
* Bsmt Cond (80 valores nulos),
* Bsmt Exposure (83 valores nulos),
* BsmtFin Type 1 (80 valores nulos),
* BsmtFin SF 1 (1 valor nulo),
* BsmtFin Type 2 (81 valores nulos),
* BsmtFin SF 2 (1 valor nulo),
* Bsmt Unf SF (1 valor nulo),
* Total Bsmt SF (1 valor nulo),
* Electrical (1 valor nulo),
* Bsmt Full Bath (2 valores nulos),
* Bsmt Half Bath (2 valores nulos),
* Fireplace Qu (1422 valores nulos),
* Garage Type (157 valores nulos),
* Garage Yr Blt (159 valores nulos),
* Garage Finish (159 valores nulos),
* Garage Cars (1 valor nulo),
* Garage Area (1 valor nulo),
* Garage Qual (159 valores nulos),
* Garage Cond (159 valores nulos),
* Pool QC (2917 valores nulos),
* Fence (2358 valores nulos),
* Misc Feature (2824 valores nulos).

Para tratar esses valores nulos, utilizei as seguintes estratégias:

* Remoção das variáveis com muitos valores nulos (mais de 50% dos dados ausentes): Mas Vnr Type, Fireplace Qu, Pool QC, Fence e Misc Feature.
* Preenchimento dos valores nulos em variáveis numéricas com a mediana da coluna.
* Preenchimento dos valores nulos em variáveis categóricas com a moda da coluna.

=== "Code"

    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    df = pd.read_csv('data/AmesHousing.csv')

    #tratamento de valores nulos
    maiores_valores_nulos = ['Pool QC', 'Misc Feature', 'Alley', 'Fence', 'Fireplace Qu']
    df = df.drop(columns=maiores_valores_nulos)
    
    # Preenchimento de valores nulos em variáveis numéricas com a mediana
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    # Preenchimento de valores nulos em variáveis categóricas com a moda
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    nulos = df.isnull().sum()
    print("\nValores ausentes por coluna:")
    print(nulos.head(100).to_string())  # Verificando se ainda há valores nulos
    ```

=== "Output"
    ```
    Valores ausentes por coluna:
        Order              0
        PID                0
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
    ```

Após tratar os valores nulos, precisamos transformar as variáveis categóricas em numéricas para que o modelo de árvore de decisão possa utilizá-las. Utilizei duas técnicas principais para isso:

**Label Encoding** -> serve para variáveis categóricas que possuem uma ordem ou hierarquia e que têm um número limitado de categorias. Exemplo: Poor, Fair, Average, Good, Excellent.

**One-Hot Encoding** -> serve para variáveis categóricas que não possuem uma ordem ou hierarquia e têm um número elevado de categorias. Exemplo: Cor do carro (vermelho, azul, verde).

**Mapeamento Manual** -> para variáveis ordinais específicas, onde atribuímos valores numéricos com base na ordem percebida.

Na tabela abaixo estão listadas as variáveis que foram transformadas utilizando cada técnica:

| Técnica            | Variáveis                                                                                       |
|--------------------|------------------------------------------------------------------------------------------------|
| Mapeamento Manual   | Exter Qual, Exter Cond, Bsmt Qual, Bsmt Cond, Heating QC, Kitchen Qual, Garage Qual, Garage Cond, Bsmt Exposure, BsmtFin Type 1, BsmtFin Type 2, Functional |
| One-Hot Encoding   | MS Zoning, Street, Lot Shape, Land Contour, Utilities, Lot Config, Land Slope, Neighborhood, Condition 1, Condition 2, Bldg Type, House Style, Roof Style, Roof Matl, Exterior 1st, Exterior 2nd, Mas Vnr Type, Foundation, Heating, Central Air, Electrical, Garage Type, Garage Finish, Paved Drive, Sale Type, Sale Condition |

* Variáveis categóricas foram transformadas em numéricas para uso no modelo.

=== "Output"
    ```
        MS SubClass  Lot Frontage  ...  Sale Condition_Normal  Sale Condition_Partial
    0           20         141.0  ...                   True                   False
    1           20          80.0  ...                   True                   False
    2           20          81.0  ...                   True                   False
    3           20          93.0  ...                   True                   False
    4           60          74.0  ...                   True                   False
    5           60          78.0  ...                   True                   False
    6          120          41.0  ...                   True                   False
    7          120          43.0  ...                   True                   False
    8          120          39.0  ...                   True                   False
    9           60          60.0  ...                   True                   False

    [10 rows x 233 columns]
    ```

=== "Code"

    ```python
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

    df = pd.get_dummies(df, columns=variaveis_nominais, drop_first=False)
    print(df.head(10))

    ```

---

## 3. Divisão dos Dados

Nessa etapa, eu criei a variável target e escolhi como features todas as outras variáveis para a construção do meu modelo. Os dados foram divididos em conjuntos de treino e teste, utilizando 70% dos dados para treino e 30% para teste. É importante garantir que a divisão seja feita de forma aleatória para evitar viés.

=== "Output"
    ```
        Target
    0    981
    1    980
    2    969
    Name: count, dtype: int64
    Features usadas no modelo:
    ['MS SubClass', 'Lot Frontage', 'Lot Area', 'Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add', 'Mas Vnr Area', 'Exter Qual', 'Exter Cond', 'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin SF 1', 'BsmtFin Type 2', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF', 'Heating QC', '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF', 'Gr Liv Area', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Full Bath', 'Half Bath', 'Bedroom AbvGr', 'Kitchen AbvGr', 'Kitchen Qual', 'TotRms AbvGrd', 'Functional', 'Fireplaces', 'Garage Yr Blt', 'Garage Cars', 'Garage Area', 'Garage Qual', 'Garage Cond', 'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch', '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Misc Val', 'Mo Sold', 'Yr Sold', 'MS Zoning_A (agr)', 'MS Zoning_C (all)', 'MS Zoning_FV', 'MS Zoning_I (all)', 'MS Zoning_RH', 'MS Zoning_RL', 'MS Zoning_RM', 'Street_Grvl', 'Street_Pave', 'Lot Shape_IR1', 'Lot Shape_IR2', 'Lot Shape_IR3', 'Lot Shape_Reg', 'Land Contour_Bnk', 'Land Contour_HLS', 'Land Contour_Low', 'Land Contour_Lvl', 'Utilities_AllPub', 'Utilities_NoSeWa', 'Utilities_NoSewr', 'Lot Config_Corner', 'Lot Config_CulDSac', 'Lot Config_FR2', 'Lot Config_FR3', 'Lot Config_Inside', 'Land Slope_Gtl', 'Land Slope_Mod', 'Land Slope_Sev', 'Neighborhood_Blmngtn', 'Neighborhood_Blueste', 'Neighborhood_BrDale', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor', 'Neighborhood_Edwards', 'Neighborhood_Gilbert', 'Neighborhood_Greens', 'Neighborhood_GrnHill', 'Neighborhood_IDOTRR', 'Neighborhood_Landmrk', 'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NAmes', 'Neighborhood_NPkVill', 'Neighborhood_NWAmes', 'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'Neighborhood_OldTown', 'Neighborhood_SWISU', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW', 'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Neighborhood_Timber', 'Neighborhood_Veenker', 'Condition 1_Artery', 'Condition 1_Feedr', 'Condition 1_Norm', 'Condition 1_PosA', 'Condition 1_PosN', 'Condition 1_RRAe', 'Condition 1_RRAn', 'Condition 1_RRNe', 'Condition 1_RRNn', 'Condition 2_Artery', 'Condition 2_Feedr', 'Condition 2_Norm', 'Condition 2_PosA', 'Condition 2_PosN', 'Condition 2_RRAe', 'Condition 2_RRAn', 'Condition 2_RRNn', 'Bldg Type_1Fam', 'Bldg Type_2fmCon', 'Bldg Type_Duplex', 'Bldg Type_Twnhs', 'Bldg Type_TwnhsE', 'House Style_1.5Fin', 'House Style_1.5Unf', 'House Style_1Story', 'House Style_2.5Fin', 'House Style_2.5Unf', 'House Style_2Story', 'House Style_SFoyer', 'House Style_SLvl', 'Roof Style_Flat', 'Roof Style_Gable', 'Roof Style_Gambrel', 'Roof Style_Hip', 'Roof Style_Mansard', 'Roof Style_Shed', 'Roof Matl_ClyTile', 'Roof Matl_CompShg', 'Roof Matl_Membran', 'Roof Matl_Metal', 'Roof Matl_Roll', 'Roof Matl_Tar&Grv', 'Roof Matl_WdShake', 'Roof Matl_WdShngl', 'Exterior 1st_AsbShng', 'Exterior 1st_AsphShn', 'Exterior 1st_BrkComm', 'Exterior 1st_BrkFace', 'Exterior 1st_CBlock', 'Exterior 1st_CemntBd', 'Exterior 1st_HdBoard', 'Exterior 1st_ImStucc', 'Exterior 1st_MetalSd', 'Exterior 1st_Plywood', 'Exterior 1st_PreCast', 'Exterior 1st_Stone', 'Exterior 1st_Stucco', 'Exterior 1st_VinylSd', 'Exterior 1st_Wd Sdng', 'Exterior 1st_WdShing', 'Exterior 2nd_AsbShng', 'Exterior 2nd_AsphShn', 'Exterior 2nd_Brk Cmn', 'Exterior 2nd_BrkFace', 'Exterior 2nd_CBlock', 'Exterior 2nd_CmentBd', 'Exterior 2nd_HdBoard', 'Exterior 2nd_ImStucc', 'Exterior 2nd_MetalSd', 'Exterior 2nd_Other', 'Exterior 2nd_Plywood', 'Exterior 2nd_PreCast', 'Exterior 2nd_Stone', 'Exterior 2nd_Stucco', 'Exterior 2nd_VinylSd', 'Exterior 2nd_Wd Sdng', 'Exterior 2nd_Wd Shng', 'Mas Vnr Type_BrkCmn', 'Mas Vnr Type_BrkFace', 'Mas Vnr Type_CBlock', 'Mas Vnr Type_Stone', 'Foundation_BrkTil', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood', 'Heating_Floor', 'Heating_GasA', 'Heating_GasW', 'Heating_Grav', 'Heating_OthW', 'Heating_Wall', 'Central Air_N', 'Central Air_Y', 'Electrical_FuseA', 'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_Mix', 'Electrical_SBrkr', 'Garage Type_2Types', 'Garage Type_Attchd', 'Garage Type_Basment', 'Garage Type_BuiltIn', 'Garage Type_CarPort', 'Garage Type_Detchd', 'Garage Finish_Fin', 'Garage Finish_RFn', 'Garage Finish_Unf', 'Paved Drive_N', 'Paved Drive_P', 'Paved Drive_Y', 'Sale Type_COD', 'Sale Type_CWD', 'Sale Type_Con', 'Sale Type_ConLD', 'Sale Type_ConLI', 'Sale Type_ConLw', 'Sale Type_New', 'Sale Type_Oth', 'Sale Type_VWD', 'Sale Type_WD ', 'Sale Condition_Abnorml', 'Sale Condition_AdjLand', 'Sale Condition_Alloca', 'Sale Condition_Family', 'Sale Condition_Normal', 'Sale Condition_Partial']
    
    Tamanho do treino: (2051, 232)
    Tamanho do teste: (879, 232)

    Distribuição das classes no target:
    Target
    1    0.344222
    0    0.338859
    2    0.316919
    Name: proportion, dtype: float64
    ```

=== "Code"

    ```python
    #criando a target para classificar o preço das casas em baixa, média e alta
    print(df['SalePrice'].describe()) 

    df['Target'] = pd.qcut( 
        df['SalePrice'], 
        q=3, 
        labels=['Baixa', 'Média', 'Alta']
    )

    df['Target'] = df['Target'].map({'Baixa':0, 'Média':1, 'Alta':2}).astype('int')

    print(df['Target'].value_counts()) 

    x = df.drop(columns=['SalePrice', 'Target'])
    y = df['Target']

    features = x.columns.tolist()
    print("Features usadas no modelo:\n", features)
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    print("Tamanho do treino:", X_train.shape)
    print("Tamanho do teste:", X_test.shape)
    print("\nDistribuição das classes no target:")
    print(y_train.value_counts(normalize=True))
    ```

=== "Explicação"

    * As variáveis de entrada (features -> variáveis x) e saída (target -> variável y) foram definidas.

    * Split 70/30 para treino e teste, garantindo avaliação justa do modelo.

---

## 4. Treinamento do Modelo

O modelo de árvore de decisão foi treinado com os dados de treino.

=== "Code"

    ```python
    #criação e treino do modelo de árvore de decisão
    clf = DecisionTreeClassifier(random_state=42, max_depth=3, criterion='entropy')  # você pode ajustar max_depth
    clf.fit(X_train, y_train)
    ```

=== "Explicação"

    * Como critério de divisão, utilizou-se a entropia para medir a incerteza com base na teoria da informação (usado quando se quer interpretar a árvore em termos de bits de informação).

    * Utilizou-se o `DecisionTreeClassifier` com profundidade máxima de 3 para evitar overfitting.

    * O modelo foi ajustado aos dados de treino.

---

## 5. Avaliação do Modelo

O desempenho do modelo foi avaliado com métricas de classificação e visualização da árvore.

=== "Output"

    ```
    Relatório de classificação:
                precision    recall  f1-score   support

            0       0.78      0.78      0.78       286
            1       0.64      0.61      0.62       274
            2       0.85      0.88      0.86       319

        accuracy                        0.76       879
    macro avg       0.75      0.76      0.75       879
    weighted avg    0.76      0.76      0.76       879
    ```

=== "Code"

    ```python
    #fazer previsões no conjunto de teste
    y_pred = clf.predict(X_test)

    #avaliação do modelo
    print("Relatório de classificação:\n", classification_report(y_test, y_pred))

    #visualização da árvore de decisão
    plt.figure(figsize=(16,8))
    plot_tree(clf, filled=True, fontsize=8, class_names=['Baixa','Média','Alta'])
    plt.savefig('./docs/classes/arvore-de-decisao/img/arvore_decisao.png')
    ```

=== "Gráfico"
    ![Árvore de Decisão](../../arvore-de-decisao/img/arvore_decisao.png)

=== "Explicação"

    * O modelo foi avaliado por métricas como precisão, recall e F1-score.

    * A acurácia geral do modelo foi de 76%, indicando um bom desempenho na classificação das casas em baixa, média e alta renda.

    * A precisão e recall foram particularmente altos para a classe de alta renda, sugerindo que o modelo é eficaz em identificar casas de maior valor.

    * O F1-score para a classe de renda média foi o mais baixo, refletindo a dificuldade em distinguir propriedades com características intermediárias.

---

O modelo de árvore de decisão desenvolvido para classificar as casas da base AmesHousing em três categorias — baixa, média e alta renda — apresentou resultados bastante satisfatórios e consistentes. A acurácia global obtida foi de 76%, indicando que o modelo conseguiu aprender de forma eficiente os padrões que distinguem as faixas de preço dos imóveis a partir de suas características estruturais, de qualidade e localização.

A análise das métricas por classe mostra um comportamento coerente com a natureza do problema. As casas de alta renda foram as mais bem classificadas, com precision de 0,85 e recall de 0,88, evidenciando que o modelo identifica de forma muito precisa os imóveis de maior valor. As casas de baixa renda também apresentaram um bom desempenho, com métricas equilibradas de precision e recall em torno de 0,78. Já a faixa de renda média foi a mais desafiadora, com F1-score de 0,62, o que é esperado, pois essas propriedades possuem características intermediárias que se sobrepõem às das outras classes — tornando sua separação menos evidente.

A estrutura da árvore de decisão revela divisões bastante intuitivas. Os principais critérios de separação estão relacionados a variáveis como qualidade geral do imóvel (Overall Qual), área habitável (Gr Liv Area) e ano de construção (Year Built), fatores que historicamente são determinantes para o valor de mercado de uma casa. O modelo dividiu os dados de forma lógica: imóveis com menor qualidade e área ficaram concentrados nos nós que levam à classificação de baixa renda, enquanto casas de padrão superior e maior metragem foram alocadas nos ramos associados à alta renda. As subdivisões intermediárias formam os grupos de renda média, onde a entropia é mais alta, refletindo a maior mistura entre categorias.

## 6. Relatório Final

Concluo que, visualmente, observa-se que a árvore possui nós bem definidos e relativamente puros, principalmente nas extremidades — o que reforça a boa capacidade de discriminação do modelo. O uso do critério de entropia permitiu identificar divisões que maximizam o ganho de informação, resultando em agrupamentos coerentes e interpretáveis. A predominância de variáveis de qualidade e tamanho nas decisões internas indica que o modelo está de fato capturando a lógica do mercado imobiliário, em que o preço é fortemente determinado por essas dimensões.

O modelo se mostra robusto, interpretável e eficiente para o objetivo proposto. Apesar de apresentar certa dificuldade na distinção entre as faixas médias de preço — algo comum em problemas desse tipo —, o desempenho global é satisfatório e demonstra que a árvore de decisão conseguiu aprender padrões reais e relevantes presentes nos dados.
