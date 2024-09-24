import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from functools import partial

# Definición de parámetros del problema
numCiudades = 10  
tamañoPoblacion = 300  
numGeneraciones = 400  
probCruce = 0.7  
probMutacion = 0.2  

# Función para generar una matriz de distancias simétrica entre ciudades
def generarMatrizDist(n):
    matriz = np.random.randint(1, 100, size=(n, n))  
    matriz = (matriz + matriz.T) / 2  
    np.fill_diagonal(matriz, 0) 
    return matriz

# Definición del problema como una minimización
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individuo", list, fitness=creator.FitnessMin)

# Función para crear un individuo como una permutación aleatoria de las ciudades
def crearIndividuo(ciudades):
    return random.sample(ciudades, len(ciudades))

# Función para evaluar el recorrido: calcula la distancia total de la ruta
def evaluarTSP(ruta, distancias):
    total = sum(distancias[ruta[i]][ruta[i+1]] for i in range(len(ruta)-1)) 
    total += distancias[ruta[-1]][ruta[0]]  
    return total,

# Configuración del algoritmo genético
toolbox = base.Toolbox()

# Función principal para ejecutar el algoritmo genético
def ejecutarAlgrtm():
    distancias = generarMatrizDist(numCiudades)  
    coordenadas = np.random.rand(numCiudades, 2) * 100  
    
    # Crear la lista de ciudades y registrar la función para generar individuos
    listCiudades = list(range(numCiudades))
    toolbox.register("individuo", tools.initIterate, creator.Individuo, partial(crearIndividuo, listCiudades))
    
    # Registro de funciones del algoritmo genético
    toolbox.register("poblacion", tools.initRepeat, list, toolbox.individuo)
    toolbox.register("mate", tools.cxTwoPoint)  
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)  
    toolbox.register("select", tools.selTournament, tournsize=3) 
    toolbox.register("evaluate", evaluarTSP, distancias=distancias)  
    
    poblacion = toolbox.poblacion(n=tamañoPoblacion) 
    result = algorithms.eaSimple(poblacion, toolbox, cxpb=probCruce, mutpb=probMutacion, ngen=numGeneraciones, verbose=False)  # Ejecutar el algoritmo genético
    
    mejor = tools.selBest(poblacion, k=1)[0]
    print(f"Mejor recorrido: {mejor}")
    print(f"Distancia mínima: {evaluarTSP(mejor, distancias)[0]}")
    
    graficar(mejor, coordenadas)

# Función para graficar el recorrido óptimo
def graficar(mejor, coordenadas):
    plt.figure(figsize=(10, 6)) 
    for i, coord in enumerate(coordenadas):
        plt.scatter(*coord, color='blue', s=100) 
        plt.text(coord[0]+1, coord[1], f'Ciudad {i}', fontsize=12, color='darkred', weight='bold')  
    
    # Graficar líneas entre ciudades en el recorrido
    for i in range(len(mejor)-1):
        inicio, fin = mejor[i], mejor[i+1]
        plt.plot([coordenadas[inicio][0], coordenadas[fin][0]], [coordenadas[inicio][1], coordenadas[fin][1]], 'g-', linewidth=2)
    
    # Cerrar el ciclo volviendo al inicio
    plt.plot([coordenadas[mejor[-1]][0], coordenadas[mejor[0]][0]], [coordenadas[mejor[-1]][1], coordenadas[mejor[0]][1]], 'g-', linewidth=2)
    plt.scatter(*coordenadas[mejor[0]], color='red', s=200, label='Inicio/Fin')  
    plt.title("Recorrido óptimo", fontsize=14, weight='bold')  
    plt.xlabel("Coordenada X")  
    plt.ylabel("Coordenada Y")  
    plt.grid(True) 
    plt.legend()  
    plt.show()  

# Función que ejecuta el algoritmo
ejecutarAlgrtm()