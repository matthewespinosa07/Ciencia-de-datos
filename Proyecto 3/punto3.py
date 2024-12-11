import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Cargar los datos
df_train = pd.read_csv("Titanic-Dataset.csv")

def preprocess_data(data):
    # Manejo de valores nulos
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

    # Codificación de variables categóricas
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    # Eliminar columnas irrelevantes
    data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    return data

df_train = preprocess_data(df_train)

# Separar características y etiquetas
X = df_train.drop('Survived', axis=1)
y = df_train['Survived']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo 1: Regresión Logística
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)

# Modelo 2: Árbol de Decisión
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

# Modelo 3: Random Forest
model_rf = RandomForestClassifier(n_estimators=100)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# Comparar resultados
print("Accuracy - Logistic Regression:", acc_lr)
print("Accuracy - Decision Tree:", acc_dt)
print("Accuracy - Random Forest:", acc_rf)

print("\nClassification Report - Logistic Regression:\n", classification_report(y_test, y_pred_lr))
print("\nClassification Report - Decision Tree:\n", classification_report(y_test, y_pred_dt))
print("\nClassification Report - Random Forest:\n", classification_report(y_test, y_pred_rf))

# Visualizar resultados
models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accuracies = [acc_lr, acc_dt, acc_rf]

plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.show()
