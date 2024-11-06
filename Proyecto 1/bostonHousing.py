import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# Dataset de Boston Housing
boston = fetch_openml(name='boston', version=1, as_frame=True)

# Datos de características (x) y objetivo (y)
x = boston.data  
y = boston.target.values 

# RM = número de habitaciones 
xRM = x['RM'].values.reshape(-1, 1) 

# Parámetros
m = 0  # Pendiente
b = 0  # Intercepto
L = 0.01  # Tasa de aprendizaje
epochs = 1000  # Número de iteraciones

n = float(len(xRM))  # Número de datos

# Algoritmo de gradiente descendente
for i in range(epochs):
    yPred = m * xRM + b  # Predicción actual
    error = y - yPred.flatten()  # Error actual, aplanar yPred para que tenga la misma dimensión que y
    derivadaM = (-2/n) * np.dot(xRM.T, error)  
    derivadaB = (-2/n) * np.sum(error)  
    m = m - L * derivadaM  
    b = b - L * derivadaB  

# Predicción final con los parámetros ajustados
yPred = m * xRM + b

# Visualización de la regresión lineal
plt.scatter(xRM, y, color='blue')  # Gráfico de dispersión de los datos reales
plt.plot(xRM, yPred, color='red')  # Línea de regresión
plt.xlabel('Número de habitaciones (RM)')
plt.ylabel('Valor medio de las casas (MEDV)')
plt.title('Regresión lineal de Boston housing')
plt.show()
