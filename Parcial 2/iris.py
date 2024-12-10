# Librerías importadas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset Iris
iris = sns.load_dataset('iris')

# Gráfico de dispersión: Longitud vs. Anchura del sépalo
plt.figure(figsize=(10, 6))
plt.scatter(iris['sepal_length'], iris['sepal_width'], c=iris['species'].astype('category').cat.codes, cmap='viridis', edgecolor='k', s=100)
plt.title('Gráfico de dispersión: Longitud vs. Anchura del sépalo')
plt.xlabel('Longitud del sépalo')
plt.ylabel('Anchura del sépalo')
plt.colorbar(label='Especie')
plt.show()
print("")

# Histograma de la longitud del sépalo con KDE
plt.figure(figsize=(10, 6))
sns.histplot(iris['sepal_length'], kde=True, bins=30)
plt.title('Distribución de la longitud del sépalo')
plt.xlabel('Longitud del sépalo')
plt.ylabel('Frecuencia')
plt.show()
print("")

# Gráfico de pares
sns.pairplot(iris, hue='species')
plt.suptitle('Gráfico de pares del dataset de Iris', y=1.02)
plt.show()
print("")

# Mapa de calor de la correlación
numeric_columns = iris.select_dtypes(include=['float64', 'int64'])

# Calcular la correlación 
correlation = numeric_columns.corr()

# Generar el mapa de calor
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Mapa de calor de correlación')
plt.show()
