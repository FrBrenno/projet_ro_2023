import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import random

""" PARAMETERS  """

# MAP PARAMETERS
MAP_DIMENSION = (70, 170)
COST_MAP_PATH = "Cost_map.txt"
PRODUCTION_MAP_PATH = "Production_map.txt"
USAGE_MAP_PATH = "Usage_map.txt"
MAP_COST_RATIO = 10000
BUDGET = 500000
random.seed(1)

# TEST MAP PARAMETERS
TEST_MAP_PATH = "20x20_"
TEST_MAP_DIMENSION = (20, 20)

# CONFIGURATION
USING_TEST_MAP = False
USING_PLOT = True

""" HELPER FUNCTIONS """


def load_usage_map(file_path):
    """Load the usage map into memory

    Args:
        file_path (str): path to the usage map file
    Information:
        0 is for a empty parcel; 1 is for a road parcel; 2 is for inhabited parcel
    """
    map_dimension = TEST_MAP_DIMENSION if USING_TEST_MAP else MAP_DIMENSION
    path = "./data/" + TEST_MAP_PATH + file_path if USING_TEST_MAP else "./data/" + file_path
    # Create an empty matrix
    matrix = np.zeros(map_dimension, dtype=int)
    # Load the file
    with open(path) as file:
        line_nb = 0
        for line in file.readlines():
            for element_nb in range(len(line) - 1):
                element = line[element_nb]
                if element == '' or element == ' ':
                    # Parcelle Inoccupée
                    matrix[line_nb, element_nb] = 0
                elif element == 'R':
                    # Parcelle occupée par une route
                    matrix[line_nb, element_nb] = 1
                elif element == 'C':
                    # Parcelle habitée
                    matrix[line_nb, element_nb] = 2
                else:
                    break
            line_nb += 1
    return matrix


def load_map(file_path):
    """Load maps into memory

    Args:
        file_path (str): path to the map file
    """
    map_dimension = TEST_MAP_DIMENSION if USING_TEST_MAP else MAP_DIMENSION
    path = "./data/" + TEST_MAP_PATH + file_path if USING_TEST_MAP else "./data/" + file_path
    # Load the file
    arr = np.loadtxt(path, dtype='str')
    # Create an empty matrix
    matrix = np.zeros(map_dimension, dtype=int)
    # Loop over each row and character to populate the matrix
    for i in range(map_dimension[0]):
        for j in range(map_dimension[1]):
            matrix[i, j] = int(arr[i][j])
    return matrix


def configure_plot(cost_map, production_map, usage_map, distance_map):
    """ Configure a figure where the matrix data is plotted.
    """
    # Create a figure with three subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    fig.canvas.manager.set_window_title("Matrix Data Plot")
    # Plot each matrix in a different subplot
    axs[0][0].set_title("Cost map")
    axs[0][0].imshow(cost_map, cmap='inferno', interpolation='nearest')  # Higher the costs are, the more Yellow it is
    axs[0][1].set_title("Production map")
    axs[0][1].imshow(production_map, cmap='Greens',
                     interpolation='nearest')  # Higher the productivity is, darker the green is
    axs[1][0].set_title("Usage map")
    axs[1][0].imshow(usage_map, cmap='gray', interpolation='nearest')
    axs[1][1].set_title("Distance map")
    axs[1][1].imshow(distance_map, cmap='Blues',
                     interpolation='nearest')  # Higher the distance is, darker the parcel is


def plot_solution(solution):
    print(solution)
    bought_plot = np.zeros(MAP_DIMENSION)
    for i in range(len(solution)):
        bought_plot[solution[i]] = 1
    plt.imshow(bought_plot, cmap='gray', interpolation='nearest')
    plt.show()


def matrice_dist(usage_matrice):
    # Trouver l'indice de tous les éléments correspondant à des habitations
    idx_habitations = np.argwhere(usage_matrice == 2)
    # trouver les distances euclidiennes entre chaque parcelle et les parcelles avec une valeur de 2 dans la matrice
    distances = np.min(cdist(np.argwhere(usage_matrice != 2), idx_habitations), axis=1)

    # reconstruire la matrice avec les distances
    distances_mat = np.zeros_like(usage_matrice, dtype=float)
    distances_mat[usage_matrice != 2] = distances
    return distances_mat


""" GENETIC ALGORITHM FUNCTIONS """


def solution_generator(cost_map, usage_map):
    bought_plot = []
    # vérifie le buget
    while sum(cost_map[bought_plot[i]] for i in range(len(bought_plot) - 1)) * 10000 < BUDGET:
        # obtenir un index aléatoire dans le tableau aplati
        new_plot_flat_index = np.random.choice(cost_map.size)
        # convertir l'index aplati en indices de ligne et de colonne
        new_plot_index = np.unravel_index(new_plot_flat_index, cost_map.shape)
        # vérifie que c'est pas une route ni une habitation et qu'on a pas déja acheté la parcelle
        if usage_map[new_plot_index] == 0 and new_plot_index not in bought_plot:
            # ajoute l'index de la parcelle à la liste des parcelles achetées
            bought_plot.append(new_plot_index)
            # vérifie que le budget n'est pas dépassé
            if sum(cost_map[bought_plot[i]] for i in range(len(bought_plot) - 1)) * 10000 > BUDGET:
                # enlève la parcelle derrnière parcelle ajouté de la liste des parcelles achetées
                bought_plot.pop()
                break
    # print(bought_plot)
    return bought_plot


def population_generator(population_size, cost_map, usage_map):
    generation = []
    for i in range(population_size):
        generation.append(solution_generator(cost_map, usage_map))

    return generation


def reproduction(parent1, parent2):
    # élimine les doublons
    for sol in parent1:
        if sol in parent2:
            parent2.remove(sol)
    # fusionne les deux parents en coupant les deux parents à un endroit aléatoire
    section = np.random.randint(0, min(len(parent1), len(parent2)))
    child1 = parent1[:section] + parent2[section:]
    child2 = parent2[:section] + parent1[section:]
    return child1, child2


def mutation(solution):
    variable_aléatoire = random.randint(0, 100)
    if variable_aléatoire <= 10:
        # enlève une parcelle aléatoire
        solution.pop(np.random.randint(0, len(solution)))
        # ajoute une parcelle aléatoire
        new_plot_flat_index = np.random.choice(cost_map.size)
        new_plot_index = np.unravel_index(new_plot_flat_index, cost_map.shape)
        if usage_map[new_plot_index] == 0 and new_plot_index not in solution:
            solution.append(new_plot_index)
    return solution



""" FITNESS FUNCTIONS """
def compacite(solution):
    aire = len(solution)
    perimetre = 0
    for i in range(len(solution)):
        if not solution.__contains__(((solution[i][0] + 1), solution[i][1])):
            perimetre += 1
        if not solution.__contains__(((solution[i][0] - 1), solution[i][1])):
            perimetre += 1
        if not solution.__contains__(((solution[i][0]), solution[i][1] + 1)):
            perimetre += 1
        if not solution.__contains__(((solution[i][0]), solution[i][1] - 1)):
            perimetre += 1
    return aire / perimetre


def proximite(solution, distance_map):
    return 1 / (sum(distance_map[solution[i]] for i in range(len(solution))) / len(solution))


def production(solution, production_map):
    return sum(production_map[solution[i]] for i in range(len(solution)))


""" SCORE FUNCTIONS """""
def score_separe(solution, distance_map, production_map):
    return compacite(solution), proximite(solution, distance_map), production(solution, production_map)


def population_with_normalized_score(generation, distance_map, production_map):
    generation_avec_score = []
    for i in range(len(generation)):
        generation_avec_score.append((generation[i], score_separe(generation[i], distance_map, production_map)))
    # normalise les scores
    max_compacite = max(generation_avec_score, key=lambda x: x[1][0])[1][0]
    min_compacite = min(generation_avec_score, key=lambda x: x[1][0])[1][0]
    max_proximite = max(generation_avec_score, key=lambda x: x[1][1])[1][1]
    min_proximite = min(generation_avec_score, key=lambda x: x[1][1])[1][1]
    max_production = max(generation_avec_score, key=lambda x: x[1][2])[1][2]
    min_production = min(generation_avec_score, key=lambda x: x[1][2])[1][2]

    population_with_normalized_score = []
    for i in range(len(generation_avec_score)):
        if max_compacite == min_compacite: # si la compacité est la même pour toutes les solutions évite la division
            # par 0
            population_with_normalized_score.append((generation_avec_score[i][0],
                                                1,
                                                (generation_avec_score[i][1][1] - min_proximite) / (
                                                        max_proximite - min_proximite),
                                                (generation_avec_score[i][1][2] - min_production) / (
                                                        max_production - min_production)))
        else:
            population_with_normalized_score.append((generation_avec_score[i][0],
                                                (generation_avec_score[i][1][0] - min_compacite) / (
                                                        max_compacite - min_compacite),
                                                (generation_avec_score[i][1][1] - min_proximite) / (
                                                        max_proximite - min_proximite),
                                                (generation_avec_score[i][1][2] - min_production) / (
                                                        max_production - min_production)))

    return population_with_normalized_score


def population_with_final_score(population_with_normalized_score):
    population_with_final_score = []
    for i in range(len(population_with_normalized_score)):
        population_with_final_score.append((population_with_normalized_score[i][0],
                                        population_with_normalized_score[i][1] * 0.33 + population_with_normalized_score[i][2] * 0.33 + population_with_normalized_score[i][3] * 0.33))
        print(population_with_normalized_score[i][0], population_with_normalized_score[i][1] * 0.33 + population_with_normalized_score[i][2] * 0.33 + population_with_normalized_score[i][3] * 0.33)
    return population_with_final_score



if __name__ == "__main__":
    """1: Loading the problem's maps"""
    cost_map = load_map(COST_MAP_PATH)
    ma_production_map = load_map(PRODUCTION_MAP_PATH)
    usage_map = load_usage_map(USAGE_MAP_PATH)
    distance_map = matrice_dist(usage_map)

    # plot_solution(solution_generator(cost_map, usage_map))
    # Plot the matrix data
    if USING_PLOT:
        configure_plot(cost_map, ma_production_map, usage_map, distance_map)
        plt.show()

    """2: INITIAL POPULATION """

    # Generate initial population randomly ⇾ cover as much as possible the solution space
    ma_population = population_generator(10, cost_map, usage_map)
    population_with_final_score(population_with_normalized_score(ma_population, distance_map, ma_production_map))
    sys.exit()
    # Evaluate initial population
    # TODO: Fitness function -> needs to be defined (weighted sums? weighted distance?)

    """3: EVOLUTION LOOP """

    # TODO: Set termination condition -> Nb of generation and/or (No variation of the surface below frontier ?)

    # while TERMINATION CONDITION
    #   Selection:          Fitter parents give two new solutions
    # TODO: ParentSelection Method (fitter with fitter/ fitter with worst/ ?)
    #   Reproduction:       Crossover of the population
    # TODO: Reproduction Method (combine/crossover data/merge/?)
    #   Mutation:           Each new solution has a probability to suffer a random mutation
    # TODO: Mutation Method (probability, how much? + random place)
    #   Evaluation:         Evaluate each new child solution
    # TODO: Use Fitness function
    #   Natural Selection:  Keep of the fitter first half of the population
    # TODO: Sort (is it needed to be sorted?) and eliminate the worst half of the population

    """4: Pareto Frontier"""

    # Determine the dominant solutions
    # Plot the frontier and generate csv files

    """5: MCDA: ELECTRE or PROMETHEE"""

    # Rank the solutions from the Pareto Frontier according to ELECTRE/PROMETHEE.

    """(?) 6: Variant to the problem (BONUS) (?)"""
