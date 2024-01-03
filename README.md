
# Regression-AI-Model-Practice

Nesse projeto eu finalmente utilizei uma IA treinada em Python, nesse caso eu adaptei e melhorei o [código](https://github.com/Victor-Lis/Regression-AI-Model) do projeto Regression-AI-Model, na qual apenas treina a IA e analisa sua taxa de acerto, porém ainda não é testado pelo usuário.

Sendo assim, nesse projeto atual é possível o usuário interagir com a IA.
# Desafios

- Entender a sintaxe do Python;
- Utilizar as bibliotecas: [pandas](https://pandas.pydata.org/docs/user_guide/index.html) e [sklearn](https://scikit-learn.org/stable/user_guide.html);
- Treinar IA, utilizando arquivos .csv;
- Criação e testes da IA.
# Aprendizados

Por final aprendi algumas coisas interessantes como: 
# Na prática

# Criando Data(Dados)
Nesse caso, eu estava querendo criar alguns valores x e y simples, apenas para entender como a IA funciona.

Então no caso abaixo, eu defini manualmente 3 funções que ao passar o X, resolveria o Y e printaria X e Y.
```python
// Esse é o arquivo que usei para copiar os valores para as tabelas de dados.

const f1 = (x) => console.log(`${x},${x*2+1}`)

console.log("Function 1")
console.log("x,y")

for(let x = 1; x <= 10; x++){
    f1(x)
}

console.log("")

const f2 = (x) => console.log(`${x},${x*4+1}`)

console.log("Function 2")
console.log("x,y")

for(let x = 1; x <= 100; x++){
    f2(x)
}

console.log("")

const f3 = (x) => console.log(`${x},${(x*3)+(x/2)}`)

console.log("Function 3")
console.log("x,y")

for(let x = 1; x <= 1000; x++){
    f3(x)
}
```

# IA

## Loading Data
Nas linhas baixo eu peço para meu usuário escolher qual das tabelas de dados ele vai escolher, ao fazer isso será atribuido a variável df.
```python
print()
dataType = ""
df = ""
while dataType != "1" and dataType != "2" and dataType != "3":
    dataType = input(f"Escolha uma opção: \n 1- Data1 \n 2- Data2 \n 3- Data3 \nR: ")
# Data 1;
if dataType == "1":
    df = pd.read_csv('https://raw.githubusercontent.com/Victor-Lis/Regression-AI-Model-Practice/master/data.csv')

# Data 2;
if dataType == "2":
    df = pd.read_csv('https://raw.githubusercontent.com/Victor-Lis/Regression-AI-Model-Practice/master/data2.csv')

# Data 3;
if dataType == "3":
    df = pd.read_csv('https://raw.githubusercontent.com/Victor-Lis/Regression-AI-Model-Practice/master/data3.csv')
```


## Data Preparation 
Nas tabelas "data", como são bem simples, tem apenas 2 colunas, X e Y. Sendo assim, fica bem auto-explicativo, y é igual a coluna y e x é igual ao restante das colunas, no caso só o x mesmo. 

Caso queira entender melhor como funcionam X e Y, fiz um [repositório](https://github.com/Victor-Lis/AI-Data-Analysis) apenas para explicar isso.
```python
y = df["y"]

x = df.drop("y", axis=1)
```


## Data Splitting
Nas linhas abaixo utizo a função train_test_split() para separar 80% dos dados para treino e 20% para teste.
```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
```

## Linear Regression

### Training the model
Nesse trecho eu utizo a função LinearRegression() para criar um modelo de regressão linear, então uso a funcão lr.fit() para treinar a IA utilizando os dados separados no bloco anterior.
```python
lr = LinearRegression()
lr.fit(x_train, y_train)
```

### Applying the model to make a prediction
Fazendo previsões utilizando os dados de treino e teste, após isso salvando os resultados, para depois avaliar a acurácia da IA. 
```python
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)
```

### Evaluate Model Performace
Nas linhas abaixo utilizo as funções mean_squared_error() e r2_score(), que expliquei melhor como funcionam no seguintes repositório [AI-Data-Analysis](https://github.com/Victor-Lis/AI-Data-Analysis)
```python
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)
```

```python
lr_results = pd.DataFrame(["Linear Regression", lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
print()
print("Result Analysis")
print(lr_results)
```

## Using

### Predict Function 
```python 
def predict():

    print()
    ### Getting number from user
    num = ""
    while num == "":
        num = input("Escreva um número: ")

    ### Convert the input number to a list with a single element
    new_data = pd.DataFrame([[float(num)]], columns=['x'])  # Assign the feature name 'x'

    ### Make a prediction using the trained model
    prediction = lr.predict(new_data)

    ### Print the prediction result
    print("Valor:", prediction[0])
    print()

    ### Restart
    restart = input("Recomeçar? y/n - ")
    if restart == "y":
        predict()

predict()
```
### Screenshots

![Escolhendo Data](https://github.com/Victor-Lis/Regression-AI-Model-Practice/blob/master/images/Escolhendo-Data.png)

![Utilizando Data1](https://github.com/Victor-Lis/Regression-AI-Model-Practice/blob/master/images/Testando%20IA%20-%20Data1.jpg)

![Utilizando Data2](https://github.com/Victor-Lis/Regression-AI-Model-Practice/blob/master/images/Testando%20IA%20-%20Data2.jpg)

![Utilizando Data3](https://github.com/Victor-Lis/Regression-AI-Model-Practice/blob/master/images/Testando%20IA%20-%20Data3.jpg)

## Autores

- [@Victor-Lis](https://github.com/Victor-Lis)