import pandas as pd
import numpy as np
import matplotlib.pyplot as plot 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

#1 - Leer los datos
df = pd.read_csv('cleaned-data.csv')

# Análisis descriptivo
print(df.head(5))

print(df.describe())  

# Variebles predictoras
X = df[['Fever', 'Tiredness', 'Dry-Cough', 'Difficulty-in-Breathing',
       'Sore-Throat', 'None_Sympton', 'Pains', 'Nasal-Congestion',
       'Runny-Nose', 'Diarrhea', 'None_Experiencing']]

# Variables target o a predecir
y = df['Severity_Moderate']

#Separación de subsets
x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.3,stratify=y)
 
knn = KNeighborsClassifier(n_neighbors=2,metric = 'manhattan')

# Entrenamiento de modelo
knn.fit(x_train,y_train)

# Predecir el modelo
y_pred = knn.predict(x_test)

prediccionPersona = knn.predict([[1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0]])
print(f'El paciente con los sintomas tiene severidad alta? {prediccionPersona}')

# Generamos la matriz de confusión
matriz = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión:')
#print(matriz)

#estimacion = knn.score(x_test,y_test)
estimacion = knn.score(x_test,y_test)
print(f'La efectividad del modelo fue de {estimacion}')

#Calculo la precisión del modelo
from sklearn.metrics import accuracy_score
exactitud = accuracy_score(y_test,y_pred)
print(f'Exactitud del modelo: {exactitud}')

precision = precision_score(y_test, y_pred)
print('Precisión del modelo:')
print(precision) 