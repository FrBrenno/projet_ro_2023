import copy
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

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
USING_DATA_PLOT = False
USE_EVOLUTION_LOOP = True

""" HELPER FUNCTIONS """


def load_usage_map(file_path):
    """Load the usage map into memory

    Args:
        file_path (str): path to the usage map file
    Information:
        0 is for a empty parcel; 1 is for a road parcel; 2 is for inhabited parcel
    """
    map_dimension = TEST_MAP_DIMENSION if USING_TEST_MAP else MAP_DIMENSION
    path = "./data/" + TEST_MAP_PATH + \
        file_path if USING_TEST_MAP else "./data/" + file_path
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
    path = "./data/" + TEST_MAP_PATH + \
        file_path if USING_TEST_MAP else "./data/" + file_path
    # Load the file
    arr = np.loadtxt(path, dtype='str')
    # Create an empty matrix
    matrix = np.zeros(map_dimension, dtype=int)
    # Loop over each row and character to populate the matrix
    for i in range(map_dimension[0]):
        for j in range(map_dimension[1]):
            matrix[i, j] = int(arr[i][j])
    return matrix


def configure_data_plot():
    """ Configure a figure where the matrix data is plotted.
    """
    # Create a figure with three subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    fig.canvas.manager.set_window_title("Matrix Data Plot")
    # Plot each matrix in a different subplot

    # COST MATRIX
    axs[0][0].set_title("Cost map")
    # Higher the costs are, the more Yellow it is
    axs[0][0].imshow(COST_MAP, cmap='inferno', interpolation='nearest')

    # PRODUCTION MATRIX
    axs[0][1].set_title("Production map")
    # Higher the productivity is, darker the green is
    axs[0][1].imshow(PRODUCTION_MAP, cmap='Greens',
                     interpolation='nearest')

    # USAGE MATRIX
    axs[1][0].set_title("Usage map")
    axs[1][0].imshow(USAGE_MAP, cmap='gray', interpolation='nearest')

    # DISTANCE MATRIX
    axs[1][1].set_title("Distance map")
    # Higher the distance is, darker the parcel is
    axs[1][1].imshow(DISTANCE_MAP, cmap='Blues',
                     interpolation='nearest')


def plot_solution(solution):

    bought_plot = copy.deepcopy(USAGE_MAP)
    for i in range(len(solution)):
        bought_plot[solution[i]] = 5

    fig, axs = plt.subplots(1, 1, figsize=(10, 9))
    fig.canvas.manager.set_window_title("solution Plot")
    plt.imshow(bought_plot, cmap='gray', interpolation='nearest')
    plt.show()


def plot_pareto(population):
    population_avec_score_normalise = population_with_normalized_score(
        population)
    pareto_frontier, population_avec_score_normalise = get_pareto_frontier(population_avec_score_normalise)
    
    
    liste_compacite = [population_avec_score_normalise[i][1]
                       for i in range(len(population_avec_score_normalise))]
    liste_proximite = [population_avec_score_normalise[i][2]
                       for i in range(len(population_avec_score_normalise))]
    liste_production = [population_avec_score_normalise[i][3]
                        for i in range(len(population_avec_score_normalise))]

    pareto_compacite = [pareto_frontier[i][1]
                        for i in range(len(pareto_frontier))]
    pareto_proximite = [pareto_frontier[i][2]
                        for i in range(len(pareto_frontier))]
    pareto_production = [pareto_frontier[i][3]
                         for i in range(len(pareto_frontier))]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.canvas.manager.set_window_title("Pareto Graph")

    ax.scatter([0], [0], [0], c='g')
    ax.scatter(liste_compacite, liste_proximite, liste_production, c='b')
    ax.scatter(pareto_compacite, pareto_proximite, pareto_production, c='r')
    ax.legend()
    ax.set_xlabel("Compacité")
    ax.set_xlim(0, 1)
    ax.set_ylabel("Proximité")
    ax.set_ylim(0, 1)
    ax.set_zlabel("Production")
    ax.set_zlim(0, 1)
    plt.show()

    create_2D_projection_image([liste_compacite, liste_proximite, liste_production], [
                               pareto_compacite, pareto_proximite, pareto_production])


def create_2D_projection_image(scores, pareto_scores):
    names = ["compacity", "proximity", "production"]
    for i in range(3):
        for j in range(i+1, 3):
            fig = plt.figure()
            ax2d = fig.add_subplot(111)
            ax2d.scatter(scores[j], scores[i])
            ax2d.scatter(pareto_scores[j], pareto_scores[i])
            ax2d.set_xlabel(f'{names[j]}')
            ax2d.set_ylabel(f'{names[i]}')
            fig.savefig(f'./img/2Dprojection_{names[j]}_{names[i]}.png')
            plt.close(fig)


def matrice_dist():
    map_dimension = TEST_MAP_DIMENSION if USING_TEST_MAP else MAP_DIMENSION
    # Trouver l'indice de tous les éléments correspondant à des habitations
    idx_habitations = np.argwhere(USAGE_MAP == 2)
    # trouver les distances euclidiennes entre chaque parcelle et les parcelles avec une valeur de 2 dans la matrice
    distances = np.min(
        cdist(np.argwhere(USAGE_MAP != 2), idx_habitations), axis=1)

    # reconstruire la matrice avec les distances
    distances_mat = np.zeros(map_dimension, dtype=float)
    distances_mat[USAGE_MAP != 2] = distances
    return distances_mat


def cost_bought_plot(bought_plot):
    return sum(COST_MAP[bought_plot[i]] for i in range(len(bought_plot) - 1)) * 10000


""" GENETIC ALGORITHM FUNCTIONS """


def solution_generator():
    """ Generates a random solution to the problem respecting budget and usage constraint
    """
    bought_plot = []
    # vérifie le buget
    while cost_bought_plot(bought_plot) < BUDGET:
        # obtenir un index aléatoire dans le tableau aplati
        new_plot_flat_index = np.random.choice(COST_MAP.size)
        # convertir l'index aplati en indices de ligne et de colonne
        new_plot_index = np.unravel_index(new_plot_flat_index, COST_MAP.shape)
        # vérifie que c'est pas une route ni une habitation et qu'on a pas déja acheté la parcelle
        if USAGE_MAP[new_plot_index] == 0 and new_plot_index not in bought_plot:
            # ajoute l'index de la parcelle à la liste des parcelles achetées
            bought_plot.append(new_plot_index)
            # vérifie que le budget n'est pas dépassé
            if cost_bought_plot(bought_plot) > BUDGET:
                # enlève la parcelle derrnière parcelle ajouté de la liste des parcelles achetées
                bought_plot.pop()
                break
    return bought_plot


def population_generator(population_size):
    """ Generates a set of new solutions
    """
    generation = []
    for i in range(population_size):
        generation.append(solution_generator())

    return generation


def reproduction(parent1, parent2):
    """ Combine two solutions in order to create to other child solutions

    Args:
        parent1: First solution
        parent2: Second solution
    """
    all_possible_parcels = parent1 + parent2
    # enlever les doublons
    unique_parcels = list(set(all_possible_parcels))
    # Diviser les parcelles en deux
    child1 = unique_parcels[:len(unique_parcels)//2]
    child2 = unique_parcels[len(unique_parcels)//2:]
    return child1, child2


def reproduction_population(population):
    """ Make the reproduction of a set of solutions

    Args:
        population: set of solutions
    """
    for i in range(0, len(population), 2):
        # Selection of the parents #! Is there any better selection model?
        parent1 = population[i]
        parent2 = population[i + 1]
        # Generation of two children
        child1, child2 = reproduction(parent1, parent2)
        population.append(child1)
        population.append(child2)
    return population


def mutation_population(population):
    """ Generate a mutation in the population in order to add randomless to the algorithm

    Args:
        population: set of solutions
    """
    nouvelle_solution_mutee = []
    for i in range(len(population)):
        solution_copie = copy.deepcopy(population[i])
        # Probabilité de 5% de mutation sur une solution
        variable_aleatoire = random.randint(0, 100)
        if variable_aleatoire <= 100:
            # enlève une parcelle aléatoire
            solution_copie.pop(np.random.randint(0, len(solution_copie)))
            # ajoute une parcelle aléatoire
            new_plot_flat_index = np.random.choice(COST_MAP.size)
            new_plot_index = np.unravel_index(
                new_plot_flat_index, COST_MAP.shape)
            i = 0
            # Vérification de validité de la parcelle choisie au hasard
            # Zone innocupée, pas déjà présente dans la solution et ne dépasse pas le budget
            while ((USAGE_MAP[new_plot_index] != 0) or (new_plot_index in solution_copie) or (cost_bought_plot(solution_copie)) > BUDGET) and i < 5:
                # Tant qu'on trouve pas une parcelle qui correspond aux critères
                i += 1
                new_plot_flat_index = np.random.choice(COST_MAP.size)
                new_plot_index = np.unravel_index(
                    new_plot_flat_index, COST_MAP.shape)
                solution_copie.append(new_plot_index)
            # Si le coût dépasse le budget, rejeter la parcelle #! deja testé non?
            while cost_bought_plot(solution_copie) > BUDGET:
                solution_copie.pop(np.random.randint(0, len(solution_copie)))
            nouvelle_solution_mutee.append(solution_copie)
    # ? Pas compris ce que ceci fait
    population.extend(nouvelle_solution_mutee)
    return population


def mutation_population2(population):
    pass


def selection(population, population_size):
    """ Selects the first half of the most optimal solutions in the current population
    """
    population_ac_score = population_with_final_score(population)
    # tri la population par score
    sorted_population_ac_score = sorted(
        population_ac_score, key=lambda x: x[1], reverse=False)
    # retourne la moitié de la population avec le meilleur score
    sorted_population = [sorted_population_ac_score[i][0]
                         for i in range(len(sorted_population_ac_score))]
    return sorted_population[:population_size]


def algorithme_genetic(initial_population_size, iteration):
    """ Genetic Algorithm:
    1: Initial population
    2: Evolution loop
        2.1: Reproduction
        2.2: Mutation
        2.3: Selection
    3: Optimal solution plot
    """
    initial_population = population_generator(initial_population_size)
    nouvelle_population = initial_population
    for i in tqdm(range(iteration)):
        # nouvelle_population = reproduction_population(nouvelle_population)
        nouvelle_population = mutation_population(nouvelle_population)
        nouvelle_population = selection(
            nouvelle_population, initial_population_size)
    print(" Solution Cost: € {:,}".format(
        cost_bought_plot(nouvelle_population[0])))
    plot_solution(nouvelle_population[0])
    print(" Solution Compacity: {:,}".format(
        compacite(nouvelle_population[0])))
    plot_solution(nouvelle_population[1])
    print(" Solution Compacity: {:,}".format(
        compacite(nouvelle_population[1])))
    plot_pareto(nouvelle_population)
    return nouvelle_population


""" FITNESS FUNCTIONS """

# Comme tous les critères sont a minimiser, le point idéal est le point (0, 0, 0)


def compacite(solution):
    """ Computes the inverse of mean of the euclidean distance of a bought parcel and center one.
    """
    # Trouver la parcelle du milieu
    milieuX = sum(plot[0] for plot in solution) / len(solution)
    milieuY = sum(plot[1] for plot in solution) / len(solution)
    # L'inverse afin d'avoir un critère à minimiser
    return sum((plot[0] - milieuX) ** 2 + (plot[1] - milieuY) ** 2 for plot in solution)


def proximite(solution):
    """ Computes the mean of the euclidian distance of a bought parcel and inhabited zone
    """
    # Critère à minimiser
    return sum(DISTANCE_MAP[solution[i]] for i in range(len(solution))) / len(solution)


def production(solution):
    """ Computes the inverse of the sum of production for each bought parcel
    """
    # Critère à minimiser
    return 1 / sum(PRODUCTION_MAP[solution[i]] for i in range(len(solution)))


""" SCORE FUNCTIONS """


def score_separe(solution):
    """ Computes separately each criteria for a given solution
    """
    return compacite(solution), proximite(solution), production(solution)


def population_with_separate_score(population):
    """ Creates a list of elements containing a solution and each respectively criteria score.
    """
    generation_avec_score = []
    for i in range(len(population)):
        generation_avec_score.append(
            (population[i], score_separe(population[i])))
    return generation_avec_score


def population_with_normalized_score(population):
    """ Normalize the criteria scores for each solution in the population
    """
    generation_avec_score = population_with_separate_score(population)

    # Trouver l'amplitude maximale pour chaque critère
    max_compacite = max(generation_avec_score, key=lambda x: x[1][0])[1][0]
    min_compacite = min(generation_avec_score, key=lambda x: x[1][0])[1][0]
    max_proximite = max(generation_avec_score, key=lambda x: x[1][1])[1][1]
    min_proximite = min(generation_avec_score, key=lambda x: x[1][1])[1][1]
    max_production = max(generation_avec_score, key=lambda x: x[1][2])[1][2]
    min_production = min(generation_avec_score, key=lambda x: x[1][2])[1][2]

    # Normaliser les scores pour chaque critère
    population_with_normalized_score = []
    for i in range(len(generation_avec_score)):

        population_with_normalized_score.append((generation_avec_score[i][0],
                                                 (generation_avec_score[i][1][0] - min_compacite) / (
            max_compacite - min_compacite),
            (generation_avec_score[i][1][1] - min_proximite) / (
            max_proximite - min_proximite),
            (generation_avec_score[i][1][2] - min_production) / (
            max_production - min_production)))

    return population_with_normalized_score


def population_with_final_score(population):
    """ Sum up each criteria score in order to have only one final score
    """

    # L'approche utilisée ici est la somme équiponderée
    population_with_score_normalized = population_with_normalized_score(
        population)
    population_with_final_score = []
    for i in range(len(population_with_score_normalized)):
        score_global = population_with_score_normalized[i][1] * 0.34 + \
            population_with_score_normalized[i][2] * 0.33 + \
            population_with_score_normalized[i][3] * 0.33
        population_with_final_score.append(
            (population_with_score_normalized[i][0], score_global))
    return population_with_final_score


""" Pareto Frontier Functions"""


def is_dominant(solution, other_solution):
    """
    Check if a solution is dominant over another solution based on three criteria.

    Returns:
    bool: True if solution is dominant over other_solution, False otherwise.
    """
    for i in range(1, 4):
        # Comme chaque critère est a minimiser, il faut que la solution ait le score le plus petit
        #! est-ce vrai? Je vois que certains critères tend vers l'infini?
        if solution[i] <= other_solution[i]:
            return False
    return True


def get_pareto_frontier(solutions):
    """ Generates a list containing the pareto optimal solutions from solutions
    """
    pareto_frontier = []
    remaining_solutions = []
    # Enumerer chaque solution afin de ne pas comparer des solutions identiques
    for i, solution in enumerate(solutions):
        is_pareto_optimal = True
        for j, other_solution in enumerate(solutions):
            # Si les solutions sont différentes et qu'elle est dominé, alors elle n'est pas optimal
            if i != j and is_dominant(other_solution, solution):
                is_pareto_optimal = False
                remaining_solutions.append(solution)
                break
        # Si la solution est optimal, ajouter à la liste
        if is_pareto_optimal:
            pareto_frontier.append(solution)
    return pareto_frontier, remaining_solutions


if __name__ == "__main__":
    """1: Loading the problem's maps"""
    COST_MAP = load_map(COST_MAP_PATH)
    PRODUCTION_MAP = load_map(PRODUCTION_MAP_PATH)
    USAGE_MAP = load_usage_map(USAGE_MAP_PATH)
    DISTANCE_MAP = matrice_dist()

    # plot_solution(solution_generator(cost_map, usage_map))
    # Plot the matrix data
    if USING_DATA_PLOT:
        configure_data_plot()
        plt.show()

    """2: INITIAL POPULATION """

    # Generate initial population randomly ⇾ cover as much as possible the solution space

    # une_population = population_generator(200)
    # plot_pareto(une_population)
    population_amelioree = algorithme_genetic(300, 500)

    # population_reproduite = algorithme_genetic(200, 200)
    # plot_pareto(population_amelioree)
    # population_with_final_score(population_with_normalized_score(ma_population, distance_map, ma_production_map))
    # population_avec_score_normalise = population_with_final_score(population_with_normalized_score(ma_population))
    # Plot
    # plot_pareto(population_avec_score_normalise)

    """4: Pareto Frontier"""

    # Determine the dominant solutions
    # Plot the frontier and generate csv files

    """5: MCDA: ELECTRE or PROMETHEE"""

    # Rank the solutions from the Pareto Frontier according to ELECTRE/PROMETHEE.

    """(?) 6: Variant to the problem (BONUS) (?)"""
