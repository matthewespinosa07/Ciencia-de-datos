# Librerías importadas
import pandas as pd
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Función para cargar y procesar los datos
def cargar_datos(url):
    respuesta = requests.get(url)
    if respuesta.status_code == 200:
        print("Datos cargados correctamente.")
        return pd.read_csv(StringIO(respuesta.text))
    else:
        print(f"Hubo un error al cargar los datos: {respuesta.status_code}")
        exit()

# URL del dataset
url_dataset = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
datos = cargar_datos(url_dataset)

# Limpieza y preparación de los datos
datos.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)  # Eliminar columnas innecesarias
datos['Age'].fillna(datos['Age'].mean(), inplace=True)  # Rellenar los valores nulos de 'Age'
datos['Embarked'].fillna(datos['Embarked'].mode()[0], inplace=True)  # Rellenar valores nulos en 'Embarked'

# Convertir variables categóricas a numéricas
datos['Sex'] = datos['Sex'].map({'male': 1, 'female': 0})
datos['Embarked'] = datos['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Separar las características y el objetivo
X = datos.drop('Survived', axis=1)
y = datos['Survived']

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo 1: Regresión Logística
modelo_logistico = LogisticRegression(max_iter=1000)
modelo_logistico.fit(X_entrenamiento, y_entrenamiento)
predicciones_logistico = modelo_logistico.predict(X_prueba)

# Evaluación del modelo de Regresión Logística
print("\nEvaluación del modelo de Regresión Logística:")
print("Precisión:", accuracy_score(y_prueba, predicciones_logistico))
print("Reporte de clasificación:\n", classification_report(y_prueba, predicciones_logistico))

# Modelo 2: Árbol de Decisión
modelo_arbol = DecisionTreeClassifier(max_depth=5, random_state=42)
modelo_arbol.fit(X_entrenamiento, y_entrenamiento)
predicciones_arbol = modelo_arbol.predict(X_prueba)

# Evaluación del modelo de Árbol de Decisión
print("\nEvaluación del modelo de Árbol de Decisión:")
print("Precisión:", accuracy_score(y_prueba, predicciones_arbol))
print("Reporte de clasificación:\n", classification_report(y_prueba, predicciones_arbol))