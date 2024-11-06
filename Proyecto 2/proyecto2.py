#####################################################################################
# AUTOR: BRAYAN MATALLANA JOYA /SANTIAGO CARVAJAL / MATTHEW ESPINOSA                 #
# FECHA: 05/11/2024                                                                 #
# DESCRIPCION: PROYECTO, Distribucion de Medicamentos en Zonas Rurales CORTE 2      #
# ASIGNATURA: BIG DATA, CIENCIA DE DATOS - TS7A                                      #
#####################################################################################

# ZONA DE IMPORTACIONES
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Se debe instalar pip install openpyxl   para que lea archivos excel.

# Cargar el archivo excel
df = pd.read_excel('distribucion_medicamentos_rurales.xlsx')

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# 1. Histograma de la demanda de medicamentos
plt.subplot(2, 2, 1)
sns.histplot(df['Demanda_Medicamentos'], bins=10, color='skyblue')
plt.title('Distribucion de la demanda de medicamentos')
plt.xlabel('Demanda de medicamentos')
plt.ylabel('Frecuencia')

# 2. Grafico de barras - Refrigeracion disponible por comunidad
plt.subplot(2, 2, 2)
sns.countplot(data=df, x='Refrigeracion_Disponible', palette='viridis')
plt.title('Disponibilidad de refrigeracion en comunidades')
plt.xlabel('Refrigeracion disponible')
plt.ylabel('Numero de comunidades')

# 3. Diagrama de dispersion - Distancia vs Demanda de medicamentos
plt.subplot(2, 2, 3)
sns.scatterplot(data=df, x='Distancia_a_Centro_Distribucion', y='Demanda_Medicamentos', hue='Nivel_Urgencia', palette='coolwarm')
plt.title('Demanda de medicamentos vs Distancia al centro de distribucion')
plt.xlabel('Distancia al centro de distribucion (km)')
plt.ylabel('Demanda de medicamentos')

# 4. Grafico de barras - Frecuencia de entrega por nivel de urgencia
plt.subplot(2, 2, 4)
sns.countplot(data=df, x='Frecuencia_Entrega', hue='Nivel_Urgencia', palette='muted')
plt.title('Frecuencia de entrega por nivel de urgencia')
plt.xlabel('Frecuencia de entrega')
plt.ylabel('Numero de comunidades')

plt.tight_layout()
plt.show()
