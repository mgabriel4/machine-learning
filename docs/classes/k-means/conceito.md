# K-Means

O K-Means é um algoritmo de aprendizado não supervisionado usado para tarefas de clusterização. Ele agrupa dados em K clusters distintos com base na similaridade das características dos dados. O objetivo do K-Means é minimizar a variância dentro de cada cluster, ou seja, os pontos dentro de um cluster devem ser o mais semelhantes possível entre si.

## Funcionamento

1. **Inicialização**: O algoritmo começa escolhendo K centróides iniciais aleatórios a partir dos dados.

2. **Atribuição de Cluster**: Cada ponto de dado é atribuído ao cluster cujo centróide está mais próximo. Essa proximidade é geralmente medida usando a distância Euclidiana.

3. **Atualização de Centróides**: Após a atribuição, os centróides de cada cluster são recalculados como a média dos pontos atribuídos a eles.

4. **Convergência**: Os passos 2 e 3 são repetidos até que os centróides não mudem significativamente ou até que um número máximo de iterações seja alcançado.

### Vantagens do K-Means ✅

* Simplicidade: O K-Means é fácil de entender e implementar.
* Escalabilidade: Funciona bem em grandes conjuntos de dados.
* Eficiência: O algoritmo é relativamente rápido, especialmente com implementações otimizadas.

### Desvantagens do K-Means ❌

* Escolha do K: A determinação do número ideal de clusters (K) pode ser desafiadora.
* Sensibilidade a outliers: O K-Means pode ser influenciado por valores extremos.
* Forma dos Clusters: O algoritmo assume que os clusters têm formas esféricas e tamanhos semelhantes, o que nem sempre é o caso na prática.
* Inicialização Aleatória: A escolha inicial dos centróides pode afetar os resultados, levando a diferentes soluções em execuções diferentes.

### Métricas para Avaliação de Clusters

As métricas mais comuns para avaliar a qualidade dos clusters formados pelo K-Means incluem:

* **Inércia**: Mede a soma das distâncias quadráticas entre os pontos e seus respectivos centróides. Menores valores de inércia indicam clusters mais compactos.

* **Silhueta**: Avalia a qualidade dos clusters, considerando a coesão (distância média entre pontos dentro do mesmo cluster) e a separação (distância média entre pontos de diferentes clusters). Valores próximos de 1 indicam clusters bem definidos, enquanto valores próximos de -1 indicam clusters mal definidos.

* **Índice de Dunn**: Mede a razão entre a menor distância entre pontos de diferentes clusters e a maior distância dentro de um cluster. Valores mais altos indicam melhores separações entre clusters.

### Método do Cotovelo (Elbow Method)

O Método do Cotovelo é uma técnica visual usada para determinar o número ideal de clusters (K) em um conjunto de dados ao usar o algoritmo K-Means. A ideia é executar o K-Means para diferentes valores de K e plotar a inércia (soma das distâncias quadráticas entre os pontos e seus respectivos centróides) em função de K. À medida que K aumenta, a inércia tende a diminuir, pois os clusters se tornam mais específicos.

O ponto onde a taxa de diminuição da inércia começa a desacelerar significativamente é chamado de "cotovelo". Esse ponto sugere um valor apropriado para K, pois indica que adicionar mais clusters além desse ponto não resulta em uma melhoria substancial na compactação dos clusters.

```mermaid
graph TD
    A[Inércia] --> B[Cotovelo]
    B --> C[Número Ideal de Clusters (K)]
```

### Aplicações do K-Means

O K-Means é amplamente utilizado em diversas áreas, incluindo:

* **Segmentação de Clientes**: Agrupar clientes com base em comportamentos de compra semelhantes.
* **Compressão de Imagens**: Reduzir o número de cores em uma imagem, agrupando pixels semelhantes.
* **Análise de Texto**: Agrupar documentos ou textos semelhantes para facilitar a busca e a organização.
* **Detecção de Anomalias**: Identificar padrões incomuns em dados, como fraudes financeiras.
* **Agrupamento Geográfico**: Agrupar locais com base em características geográficas ou demográficas.
* **Recomendações de Produtos**: Agrupar produtos semelhantes para melhorar sistemas de recomendação.
* **Análise de Redes Sociais**: Identificar comunidades ou grupos de usuários com interesses semelhantes.
* **Biologia e Genômica**: Agrupar genes ou proteínas com funções semelhantes para facilitar a análise biológica.
* **Marketing**: Identificar segmentos de mercado para campanhas direcionadas.
* **Saúde**: Agrupar pacientes com características semelhantes para personalizar tratamentos médicos.
* **Ciência de Dados**: Explorar e entender grandes conjuntos de dados, identificando padrões e tendências.
* **Finanças**: Agrupar ativos financeiros com comportamentos semelhantes para otimizar portfólios de investimento.
* **Logística**: Otimizar rotas de entrega agrupando destinos próximos.
* **Educação**: Agrupar estudantes com base em desempenho ou estilos de aprendizagem para personalizar o ensino.
