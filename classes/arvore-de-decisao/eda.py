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

# plt.figure(figsize=(10, 8))
# correlation_matrix = data.corr(numeric_only=True)
# sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
# plt.title('Matriz de Correlação')
# plt.savefig('./docs/classes/arvore-de-decisao/img/matriz_correlacao.png')