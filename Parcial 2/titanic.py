# Librerías importadas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset Titanic
titanic = sns.load_dataset('titanic')

# Gráfico de conteo: Supervivencia por clase
plt.figure(figsize=(10, 6))
sns.countplot(data=titanic, x='class', hue='survived', palette='pastel')
plt.title('Supervivencia por clase')
plt.xlabel('Clase')
plt.ylabel('Número de pasajeros')
plt.legend(title='Supervivió', labels=['No', 'Sí'])
plt.show()
print("")

# Histograma de la edad de los sobrevivientes
plt.figure(figsize=(10, 6))
sns.histplot(data=titanic[titanic['survived'] == 1]['age'], bins=30, kde=True, color='green')
plt.title('Distribución de edad de sobrevivientes')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')
plt.show()
print("")

# Gráfico de conteo: Supervivencia por género
plt.figure(figsize=(10, 6))
sns.countplot(data=titanic, x='sex', hue='survived', palette='pastel')
plt.title('Supervivencia por género')
plt.xlabel('Género')
plt.ylabel('Número de pasajeros')
plt.legend(title='Supervivió', labels=['No', 'Sí'])
plt.show()
print("")

# Mapa de calor de la correlación
numeric_columns = titanic.select_dtypes(include=['float64', 'int64'])

# Eliminar filas con valores NaN
numeric_columns = numeric_columns.dropna()

# Calcular la correlación
correlation = numeric_columns.corr()

# Generar el mapa de calor
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Mapa de calor de correlación')
plt.show()
print("")
