# Regressão com Support Vector Machine (SVR) na base AmesHousing

## 1. Contexto e objetivo

A base **AmesHousing** contém informações detalhadas sobre imóveis na cidade de Ames (EUA), com diversas variáveis estruturais, de localização e de qualidade da construção, além do **preço de venda (`SalePrice`)**.

O objetivo deste experimento é:

> Utilizar um modelo **Support Vector Machine para regressão (SVR)** para prever o preço de venda dos imóveis com base nas demais variáveis da base.

---

## 2. EDA da base

- **Nº de observações:** 2.930 imóveis  
- **Nº de colunas originais:** 82  
- Variáveis incluem:
  - Identificadores: `Order`, `PID`
  - Estruturais: `Overall Qual`, `Overall Cond`, `Gr Liv Area`, `Total Bsmt SF` etc.
  - Categóricas: `MS Zoning`, `Neighborhood`, `Bldg Type`, `House Style` etc.
  - Variável alvo (target): **`SalePrice`** (valor de venda do imóvel, em dólares)

Resumo da variável `SalePrice`:

- **Média:** ≈ 180.796  
- **Desvio-padrão:** ≈ 79.887  
- **Mínimo:** 12.789  
- **Máximo:** 755.000  

Isso mostra uma boa variação de preços, com imóveis bem mais baratos e outros bem mais caros.

---

## 3. Preparação dos dados

### 3.1. Remoção de colunas não informativas

Foram removidas as colunas:

- `Order` – apenas ordem no dataset  
- `PID` – identificador único do imóvel

Essas variáveis não carregam informação útil para prever o preço.

### 3.2. Definição de features e target

- **Target (`y`):** `SalePrice`  
- **Features (`X`):** todas as demais colunas, exceto `Order`, `PID` e `SalePrice`.

### 3.3. Tratamento das variáveis

As variáveis categóricas foram transformadas em variáveis numéricas usando **one-hot encoding**:

```python

X_dum = pd.get_dummies(X, drop_first=True)

```

Após o one-hot:

* Nº de linhas: 2.930

* Nº de colunas: 261 (incluindo todas as dummies)

Tivemos que tratar os valores ausentes, pois o SVM (SVR) não aceita valores NaN. Para isso, foi aplicado um SimpleImputer com estratégia de mediana:

* Todas as colunas numéricas (incluindo dummies) com valores faltantes foram preenchidas com a mediana da respectiva coluna.

Isso foi feito dentro do pipeline, para evitar data leakage.

Padronizamos as features, pois como o SVM é sensível à escala das variáveis, as features foram padronizadas, utilizando o método *StandardScaler()*. Cada coluna foi transformada para ter média ≈ 0 e desvio-padrão ≈ 1.

Logo após disso, como SalePrice está em uma escala alta, foi usada a classe TransformedTargetRegressor com StandardScaler também no alvo, para facilitar o ajuste do SVR. Assim, o modelo aprende sobre o SalePrice padronizado. As previsões são, depois, transformadas de volta para o espaço original (em dólares).

## 4. Divisão treino / teste

Os dados foram divididos em: 70% treino e 30% teste.

random_state = 42 para reprodutibilidade

```python

X_train, X_test, y_train, y_test = train_test_split(
    X_dum, y, test_size=0.3, random_state=42
)
```

## 5. Modelagem com SVM(SVR)

Foi utilizado um Support Vector Regressor (SVR) com kernel RBF:

5.1. Estrutura do pipeline

```python
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf", C=100.0, epsilon=0.1))
])

model = TransformedTargetRegressor(
    regressor=pipe,
    transformer=StandardScaler()
)
```

Componentes:

1. Imputer (mediana): Trata valores ausentes em todas as colunas.

2. StandardScaler: Padroniza as features.

3. SVR (kernel RBF):
    * kernel="rbf": kernel gaussiano, captura relações não lineares.
    * C=100.0: penalização relativamente alta para erros (modelo mais “rígido”).
    * epsilon=0.1: margem de tolerância em torno das previsões no espaço padronizado.

4. TransformedTargetRegressor + StandardScaler: Escala o target SalePrice para facilitar o treino do SVR.

### 5.2. Treino do modelo

``` python
model.fit(X_train, y_train)
``` 

Após o treino, foram feitas previsões na base de teste:

``` python
y_pred = model.predict(X_test)
```

## 6. Avaliação do Modelo

|  Métrica |          Valor |
| -------: | -------------: |
|  **MSE** | 958.564.501,73 |
| **RMSE** |      30.960,69 |
|  **MAE** |      18.716,88 |
|   **R²** |         0,8636 |

### Interpretação

* O R² ≈ 0,86 indica que o modelo SVR consegue explicar cerca de 86% da variação do preço de venda dos imóveis na base de teste.

* O RMSE ≈ 30,9 mil significa que, em média, o erro quadrático médio em termos de desvio típico das previsões é da ordem de 31 mil dólares.

* O MAE ≈ 18,7 mil quer dizer que, em valor absoluto, o erro médio que o modelo comete por imóvel é de cerca de 18,7 mil dólares.