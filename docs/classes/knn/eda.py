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

# Visualização das variáveis categóricas
plt.style.use('ggplot')
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.countplot(x='target', data=dados)
plt.title('Distribuição da variável alvo')
plt.ylabel('Contagem')

plt.subplot(2, 2, 2)
sns.countplot(x='sex', data=dados)
plt.title('Distribuição por Sexo')
plt.ylabel('Contagem')

plt.subplot(2, 1, 2)
sns.countplot(x='cp', data=dados)
plt.title('Distribuição por Tipo de Dor no Peito')
plt.ylabel('Contagem')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./docs/classes/knn/img/distribuicao.png')
plt.show()

# Visualização das variáveis numéricas
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
sns.histplot(dados['age'], bins=20, kde=True)
plt.title('Distribuição da Idade')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.subplot(2, 2, 2)
sns.histplot(dados['trestbps'], bins=20, kde=True)
plt.title('Distribuição da Pressão Arterial em Repouso')
plt.xlabel('Pressão Arterial (mm Hg)')
plt.ylabel('Frequência')
plt.subplot(2, 1, 2)
sns.histplot(dados['chol'], bins=20, kde=True)
plt.title('Distribuição do Colesterol')
plt.xlabel('Colesterol (mg/dl)')
plt.ylabel('Frequência')
plt.tight_layout()
plt.savefig('./docs/classes/knn/img/distribuicao_numerica.png')
plt.show()

