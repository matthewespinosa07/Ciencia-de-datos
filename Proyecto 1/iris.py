import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
x = iris.data.features 
y = iris.data.targets 
  
# metadata 
print(f"\nCaracterísticas (X):\n{x.head(10)}")

# variable information 
print(f"\nObjetivo (Y):\n{y.head(10)}")

# Estadísticas descriptivas
media = x.mean()
mediana = x.median()
desviacionEstandar = x.std()

print(f"\nMedia:\n{media}")
print(f"\nMediana:\n{mediana}")
print(f"\nDesviación estándar:\n{desviacionEstandar}")