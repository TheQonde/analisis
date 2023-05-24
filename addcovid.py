# 1- Importar las bibliotecas
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 

# 2 - Leer los datos
df = pd.read_csv('covid.csv')

# 3 - Entender los datos
print(df.info())
print(df.sample)

# 4 - Eliminar filas con valores faltantes
df.dropna(inplace=True)

# 5 - Variables predictoras
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Convertir variables categóricas a numéricas
X = pd.get_dummies(X)

# 5 - Separar los conjuntos de datos
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6 - Definición del modelo a utilizar (Bosques Aleatorios)
rf = RandomForestClassifier()

rf.fit(x_train, y_train)

exactitudTrain = rf.score(x_train, y_train)
exactitudTest = rf.score(x_test, y_test)

print(f'La exactitud de los datos de entrenamiento es de {exactitudTrain}')
print(f'La exactitud de los datos de prueba es de {exactitudTest}')

# 6 - Predecir con el modelo
y_pred = rf.predict(x_test)

# 7 - Generar la matriz de confusión
print(f'¿El paciente con los síntomas tiene severidad alta? {y_pred}')

# Calcular la precisión del modelo
from sklearn.metrics import accuracy_score
exactitud = accuracy_score(y_test, y_pred)
print(f'Exactitud del modelo: {exactitud}')

# Importancia de las características
print("Feature importances:\n{}".format(rf.feature_importances_))

# 7 - Generar la matriz de confusión
matriz = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión:')
print(matriz)

#plt.show()
