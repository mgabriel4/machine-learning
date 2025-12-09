import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import processamento as proc

#cáculo da importância das variáveis
importances = proc.clf.feature_importances_
indices = np.argsort(importances)[::-1]

#tive que selecionar as 5 variáveis mais importantes
top_n = 5
plt.figure(figsize=(10,6))
plt.title("Importância das Variáveis")
sns.barplot(x=importances[indices][:top_n], y=np.array(proc.X_train.columns)[indices][:top_n], palette="viridis")
plt.xlabel("Importância")
plt.ylabel("Variáveis")
plt.tight_layout()
plt.savefig('./docs/classes/arvore-de-decisao/img/importancia_variaveis.png')

# Matriz de confusão
cm = confusion_matrix(proc.y_test, proc.y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Baixa', 'Média', 'Alta'],
            yticklabels=['Baixa', 'Média', 'Alta'])
plt.xlabel("Classe Prevista")
plt.ylabel("Classe Real")
plt.title("Matriz de Confusão - Árvore de Decisão")
plt.tight_layout()
plt.savefig('./docs/classes/arvore-de-decisao/img/matriz_confusao.png')