import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree 

df = pd.read_csv('data/AmesHousing.csv')

df=df.drop(columns=['Order', 'PID'])  #removendo colunas irrelevantes

#tratamento de valores nulos
maiores_valores_nulos = ['Pool QC', 'Misc Feature', 'Alley', 'Fence', 'Fireplace Qu']
df = df.drop(columns=maiores_valores_nulos)

#variáveis numéricas -> preencher com mediana
#variáveis categóricas -> preencher com moda
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

nulos = df.isnull().sum()
print("\nValores ausentes por coluna:")
print(nulos.head(100).to_string())

#label encoding para variáveis categóricas
#usamos label encoding quando a variável categórica é ordinal (tem uma ordem) e tem poucas categorias
#cada categoria recebe um número inteiro

#porém, ao invés de fazer o Label Encoding, fiz o mapeamento para variáveis ordinais, para manter a ordem
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

#one hot encoding para variáveis categóricas nominais
#usamos one hot encoding quando a variável categórica não é ordinal (não tem uma ordem) e tem poucas categorias
#cria uma nova coluna para cada categoria, com 0 ou 1 indicando a presença ou ausência da categoria

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

#criando a target para classificar o preço das casas em baixa, média e alta
print(df['SalePrice'].describe()) 

df['Target'] = pd.qcut(  #divide em quantis
    df['SalePrice'], 
    q=3, 
    labels=['Baixa', 'Média', 'Alta']
)

#transformando a variável target em numérica
df['Target'] = df['Target'].map({'Baixa':0, 'Média':1, 'Alta':2}).astype('int')

print(df['Target'].value_counts()) #o que coincide com a divisão em tercis, ou seja, 1/3 dos dados em cada classe

#features -> são as variáveis de entrada, colunas a serem usadas pra prever ou explicar algo (x)
x = df.drop(columns=['SalePrice', 'Target']) #selecionei todas as colunas menos a target e a SalePrice
y = df['Target']

features = x.columns.tolist()
print("Features usadas no modelo:\n", features)
#divisão dos dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print("Tamanho do treino:", X_train.shape)
print("Tamanho do teste:", X_test.shape)
print("\nDistribuição das classes no target:")
print(y_train.value_counts(normalize=True))

#criação e treino do modelo de árvore de decisão
clf = DecisionTreeClassifier(random_state=42, max_depth=3, criterion='entropy')  # você pode ajustar max_depth
clf.fit(X_train, y_train)

#fazer previsões no conjunto de teste
y_pred = clf.predict(X_test)

#avaliação do modelo
print("Relatório de classificação:\n", classification_report(y_test, y_pred))

#visualização da árvore de decisão
plt.figure(figsize=(16,8))
plot_tree(clf, filled=True, fontsize=8, class_names=['Baixa','Média','Alta'])
plt.savefig('./docs/classes/arvore-de-decisao/img/arvore_decisao.png')
plt.show()