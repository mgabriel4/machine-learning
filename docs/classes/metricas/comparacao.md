# Comparação entre modelos

| Modelo            | Dataset / Target                                                                      | Tipo de problema              | Métricas usadas (no texto/código)                                                       | Números registrados na doc                                      |
| ----------------- | ------------------------------------------------------------------------------------- | ----------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| Árvore de Decisão | Ames Housing → classes de preço: baixa, média, alta ([mgabriel4.github.io][1])        | Classificação (3 classes)     | Matriz de confusão, análise de erros entre classes                                      | **Nenhum valor numérico** explícito (sem acurácia/F1 na página) |
| KNN               | Ames Housing → mesmas 3 classes de preço ([mgabriel4.github.io][2])                   | Classificação (3 classes)     | Acurácia (treino/teste e cross-val), precision, recall, F1, matriz de confusão          | **CV**: acc ≈ 0,79; **teste**: acc ≈ 0,80; F1 macro ≈ 0,80      |
| K-Means           | Ames Housing → clusters interpretados como faixas de preço ([mgabriel4.github.io][3]) | Clustering não supervisionado | Inertia (Elbow curve), distribuição de clusters vs faixas de preço, análise qualitativa | Sem acurácia/F1; só contagem de Target x Cluster                |
                                         |
