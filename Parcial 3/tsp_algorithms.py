# Librerías importadas
import random
import math
from collections import defaultdict


# Algoritmo 2-SAT
def add_implication_to_graph(graph, var1, var2):
    graph[var1].append(var2)
    graph[var2 ^ 1].append(var1 ^ 1)

def depth_first_search(graph, node, visited_nodes, stack):
    visited_nodes[node] = True
    for neighbor in graph[node]:
        if not visited_nodes[neighbor]:
            depth_first_search(graph, neighbor, visited_nodes, stack)
    stack.append(node)

def solve_2sat(num_variables, clauses):
    graph = defaultdict(list)
    for var1, var2 in clauses:
        add_implication_to_graph(graph, var1, var2)
    
    visited_nodes = [False] * (2 * num_variables)
    node_stack = []
    
    for i in range(2 * num_variables):
        if not visited_nodes[i]:
            depth_first_search(graph, i, visited_nodes, node_stack)
    
    visited_nodes = [False] * (2 * num_variables)
    result = [False] * num_variables
    
    while node_stack:
        node = node_stack.pop()
        if not visited_nodes[node ^ 1]:
            visited_nodes[node] = True
            visited_nodes[node ^ 1] = True
            result[node // 2] = (node % 2 == 0)
    
    return result


# Algoritmo TSP (Traveling Salesman Problem) con 2-opt
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def compute_total_distance(tour_order, points_list):
    total = 0
    for i in range(len(tour_order)):
        total += calculate_distance(points_list[tour_order[i]], points_list[tour_order[(i + 1) % len(tour_order)]])
    return total

def optimize_with_two_opt(points_list):
    num_points = len(points_list)
    initial_tour = list(range(num_points))
    best_distance = compute_total_distance(initial_tour, points_list)
    
    improvement = True
    while improvement:
        improvement = False
        for i in range(num_points - 1):
            for j in range(i + 2, num_points):
                new_tour = initial_tour[:i + 1] + list(reversed(initial_tour[i + 1:j + 1])) + initial_tour[j + 1:]
                new_distance = compute_total_distance(new_tour, points_list)
                if new_distance < best_distance:
                    initial_tour = new_tour
                    best_distance = new_distance
                    improvement = True
    return initial_tour, best_distance


# Algoritmo de Recocido Simulado
def quadratic_energy(x):
    return x ** 2 - 4 * x + 4  # Una función cuadrática simple

def simulated_annealing_algorithm(initial_value, initial_temperature, cooling_factor, max_iterations):
    current_value = initial_value
    best_value = current_value
    current_energy = quadratic_energy(current_value)
    best_energy = current_energy

    for _ in range(max_iterations):
        new_value = current_value + random.uniform(-1, 1)
        new_energy = quadratic_energy(new_value)

        if new_energy < current_energy or random.random() < math.exp((current_energy - new_energy) / initial_temperature):
            current_value = new_value
            current_energy = new_energy

            if current_energy < best_energy:
                best_value = current_value
                best_energy = current_energy

        initial_temperature *= cooling_factor

    return best_value, best_energy


# Ejecutar los algoritmos
if __name__ == "__main__":
    # Ejemplo de 2-SAT
    print("2-SAT")
    num_vars = 3  # Tres variables
    clause_list = [(0, 1), (2, 3), (1, 2)]  # Clausulas (x1 or x2, x2 or x3, x1 or x3)
    solution = solve_2sat(num_vars, clause_list)
    print("Resultado 2-SAT:", solution)

    # Ejemplo de TSP con 2-opt
    print("\nTSP")
    coordinates = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    optimal_tour, optimal_distance = optimize_with_two_opt(coordinates)
    print("Ruta optimizada:", optimal_tour)
    print("Distancia total:", optimal_distance)

    # Ejemplo de Recocido Simulado
    print("\nRecocido Simulado")
    initial_state = random.uniform(-10, 10)
    initial_temp = 1000
    cooling_rate = 0.99
    max_iter = 1000

    best_state, best_state_energy = simulated_annealing_algorithm(initial_state, initial_temp, cooling_rate, max_iter)
    print("Mejor solución encontrada:", best_state)
    print("Valor mínimo de energía:", best_state_energy)