import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree 

data = pd.read_csv("data/BMW_Car.csv")

data['Model_Num'] = LabelEncoder().fit_transform(data['Model']) #transforma a variável Model em numérica
data['Region_Num'] = LabelEncoder().fit_transform(data['Region']) #transforma a variável Region em numérica
data['Color_Num'] = LabelEncoder().fit_transform(data['Color']) #transforma a variável Color em numérica
data['Fuel_Num'] = LabelEncoder().fit_transform(data['Fuel_Type']) #transforma a variável Fuel Type em numérica

print(data.columns.tolist())

#one hot encoding
data = pd.get_dummies(data, columns=['Transmission', 'Sales_Classification'], drop_first=True) # aplica one hot encoding

'''tratamento de valores nulos
 -> COLOCAMOS MÉDIA OU MEDIANA PARA TRATAR VARIÁVEIS NÚMERICAS 
 -> EM VARIÁVEIS CATEGÓRICAS COLOCAMOS A MODA
Se tivéssemos valores nulos nesta base, trataríamos desta forma:
for col in ['Engine_Size_L','Mileage_KM','Price_USD','Sales_Volume']:
    data[col] = data[col].fillna(data[col].median())

for col in ['Model_Num','Region_Num','Color_Num','Fuel_Num']:
    data[col] = data[col].fillna(data[col].mode()[0])
'''

#conferindo se deu certo o pré-processamento
print(data.isnull().sum())
print(data.head(10))

#features -> são as variáveis de entrada, colunas a resem usadas pra prever ou explicar algo (x)
features = ['Year','Region_Num', 'Fuel_Num', 'Transmission_Manual']

x = data[features]
y = data['Sales_Classification_Low'] #variável target -> variavel de saida, ou o que se quer prever (y)

#divisão dos dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#criação e treino do modelo de árvore de decisão
clf = DecisionTreeClassifier(random_state=42, max_depth=3)  # você pode ajustar max_depth
clf.fit(X_train, y_train)

#fazer previsões no conjunto de teste
y_pred = clf.predict(X_test)

#avaliação do modelo
print("Relatório de classificação:\n", classification_report(y_test, y_pred))

#visualização da árvore de decisão
plt.figure(figsize=(10,5))
tree = plot_tree(clf, feature_names=features, class_names=['Not Low','Low'], filled=True, rounded=True)
plt.savefig('./docs/arvore-de-decisao/img/arvorecars.png')
plt.show()