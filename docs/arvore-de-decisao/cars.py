import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#carregar o dataset
data = pd.read_csv('docs/arvore-de-decisao/data/BMW_Car.csv')

# Análise exploratória inicial
print("Primeiras 5 linhas do dataset:")
print(data.head())

print("\nInformações do dataset:")
print(data.info())

print("\nEstatísticas descritivas:")
print(data.describe(include='all'))

print("\nValores ausentes por coluna:")
print(data.isnull().sum())

#visualização das variáveis categóricas
plt.style.use('ggplot')
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
data['Transmission'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(6,6))
plt.title('Tipo de carro')
plt.ylabel('')  #remove o label do eixo Y

plt.subplot(2, 2, 2)
data['Fuel_Type'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(6,6))
plt.title('Tipo de Combustível')
plt.ylabel('')  #remove o label do eixo Y

plt.subplot(2, 1, 2)
data['Region'].value_counts().head(5).plot(kind='bar')
plt.title('Top 5 Regiões')
plt.xticks(rotation=45)

plt.savefig('./docs/arvore-de-decisao/img/distribuicao.png')
plt.show()

print(data.columns.tolist())