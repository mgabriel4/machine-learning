# Projeto Random Forest

Neste projeto, vamos construir um modelo de Random Forest para classificar as casas da base Ames Housing em diferentes categorias de preço.

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
    from matplotlib import patches
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from IPython.display import display

    df = pd.read_csv('data/AmesHousing.csv')

    print("Primeiras 5 linhas do dataset:")
    print(df.head())

    print("\nInformações do dataset:")
    print(df.info())

    print("\nEstatísticas descritivas:")
    print(df.describe(include='all'))

    colunas = df.columns
    print("\nColunas do dataset:")
    print(colunas)

    nulos = df.isnull().sum()
    print("\nValores ausentes por coluna:")
    print(nulos.head(100).to_string())

    plt.style.use('ggplot')

    plt.figure(figsize=(15,10))
    df.isnull().mean().sort_values(ascending=False).head(20).plot(kind='bar', color='salmon')
    plt.title("Percentual de Valores Ausentes nas 20 Variáveis com Mais NAs")
    plt.ylabel("% de valores faltantes")
    plt.savefig('./docs/classes/arvore-de-decisao/img/valores_ausentes.png')

    #variáveis categóricas
    plt.figure(figsize=(25, 20))
    plt.subplot(2, 2, 1)
    sns.countplot(data=df, x='MS Zoning', palette='viridis', order=df['MS Zoning'].value_counts().index)
    plt.title('Classificação de Zonas Residenciais')
    plt.xticks(rotation=45)
    plt.xlabel('Zona de Zoneamento')
    plt.ylabel('Contagem')
    plt.tight_layout()

    plt.subplot(2, 2, 2)
    df['Street'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(6,6), colors=sns.color_palette('viridis', 5), labels=df['Street'].value_counts().index)
    plt.title('Tipo de Rua')
    plt.ylabel('')  #remove o label do eixo Y
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    df['Neighborhood'].value_counts().head(5).plot(kind='bar', color=sns.color_palette('viridis', 5))
    plt.title('Top 5 Vizinhanças')
    plt.xticks(rotation=45)
    plt.xlabel('Vizinhança')
    plt.ylabel('Contagem')
    plt.tight_layout()

    plt.subplot(2, 2, 4)
    sns.countplot(data=df, x='House Style', palette='viridis', order=df['House Style'].value_counts().index)
    plt.xticks(rotation=45)
    plt.title('Estilo da Casa')
    plt.ylabel('Contagem')
    plt.tight_layout()
    plt.savefig('./docs/classes/arvore-de-decisao/img/categoricas.png')

    #variáveis numéricas
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    sns.histplot(df['SalePrice'], bins=30, kde=True, color='salmon')
    plt.title("Distribuição do Preço de Venda")
    plt.xlabel("Preço de Venda")
    plt.ylabel("Frequência")
    plt.tight_layout()

    plt.subplot(2, 2, 2)
    sns.histplot(df['Gr Liv Area'], bins=30, kde=True, color='salmon')
    plt.title("Distribuição da Área Acima do Solo")
    plt.xlabel("Área Acima do Solo (pés²)")
    plt.ylabel("Frequência")
    plt.tight_layout()

    plt.subplot(2, 2, 3)
    sns.histplot(df['Year Built'], bins=30, kde=True, color='salmon')
    plt.title("Distribuição do Ano de Construção")
    plt.xlabel("Ano de Construção")
    plt.ylabel("Frequência")
    plt.tight_layout()

    plt.subplot(2, 2, 4)
    sns.histplot(df['Total Bsmt SF'], bins=30, kde=True, color='salmon')
    plt.title("Distribuição da Área do Porão")
    plt.xlabel("Área do Porão (pés²)")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig('./docs/classes/arvore-de-decisao/img/dist_geral_numericas.png')
    plt.show()  

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    sns.boxplot(x='Neighborhood', y='SalePrice', palette='viridis', data=df)
    plt.title("Preço por Bairro")
    plt.xticks(rotation=90)

    plt.subplot(2, 1, 2)
    sns.boxplot(x='Overall Qual', y='SalePrice', data=df, palette='viridis')
    plt.title("Qualidade Geral vs Preço de Venda")

    plt.tight_layout()
    plt.savefig('./docs/classes/arvore-de-decisao/img/boxplots.png')

    corr_cols = ['Lot Area', 'Gr Liv Area', 'Total Bsmt SF', 'Garage Area', '1st Flr SF',
        '2nd Flr SF', 'Full Bath', 'Half Bath', 'Bedroom AbvGr', 'TotRms AbvGrd',
        'Overall Qual', 'Overall Cond', 'Year Built', 'Year Remod/Add']
    plt.figure(figsize=(10,8))
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlação entre variáveis numéricas")
    plt.savefig('./docs/classes/arvore-de-decisao/img/matriz_correlacao.png')
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

## Passo 2: Carregar o Conjunto de Dados

Carregamos o conjunto de dados Iris diretamente do repositório do UCI Machine Learning Repository.

    ```python
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        data = pd.read_csv(url, names=columns)
    ```

## Passo 3: Pré-processamento dos Dados

Verificamos se há valores ausentes e codificamos a variável alvo.

    ```python
        print(data.isnull().sum())
        data['species'] = data['species'].astype('category').cat.codes
    ```

## Passo 4: Dividir os Dados em Conjuntos de Treinamento e Teste

Dividimos os dados em conjuntos de treinamento e teste para avaliar o desempenho do modelo.

    ```python
    X = data.drop('species', axis=1)
    y = data['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

## Passo 5: Construir o Modelo Random Forest

Criamos e treinamos o modelo Random Forest.

    ```python
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    ```

## Passo 6: Avaliar o Modelo

Avaliamos o desempenho do modelo usando métricas como matriz de confusão e relatório de classificação.

    ```python
    y_pred = rf_model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    ```

## Conclusão

O modelo de Random Forest foi capaz de classificar as espécies de flores com uma boa precisão. A matriz de confusão e o relatório de classificação fornecem insights sobre o desempenho do modelo e suas áreas de melhoria. Com ajustes adicionais e mais dados, o modelo pode ser aprimorado ainda mais.
