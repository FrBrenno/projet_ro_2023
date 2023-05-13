import copy
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import random
from tqdm import tqdm

""" PARAMETERS  """

# MAP PARAMETERS
#MAP_DIMENSION = (70, 170)
COST_MAP_PATH = "Cost_map.txt"
PRODUCTION_MAP_PATH = "Production_map.txt"
USAGE_MAP_PATH = "Usage_map.txt"
MAP_COST_RATIO = 10000
BUDGET = 500000



# TEST MAP PARAMETERS
TEST_MAP_PATH = "20x20_"
TEST_MAP_DIMENSION = (20, 20)

# CONFIGURATION
USING_TEST_MAP = False
USING_DATA_PLOT = False
USE_EVOLUTION_LOOP = True

best_scores = []

""" HELPER FUNCTIONS """


def get_map_dimension(file_path):
    path = "./data/" + TEST_MAP_PATH + \
           file_path if USING_TEST_MAP else "./data/" + file_path
    with open(path) as file:
        line_nb = 0
        collumn_nb = 0
        for line in file.readlines():
            line_nb += 1
            if line_nb == 1:
                for element_nb in range(len(line) - 1):
                    collumn_nb += 1
    return (line_nb, collumn_nb)



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
    fig.canvas.manager.set_window_title("solution Plot:  " + "compacity: " + str(compacite(solution)) + " proximite: " + str(proximite(solution)) + " production: " + str(production(solution)))
    plt.imshow(bought_plot, cmap='gray', interpolation='nearest')
    plt.show()


def plot_pareto(population):
    population_avec_score_normalise = population_with_separate_score(
        population)
    pareto_frontier, population_avec_score_normalise = get_pareto_frontier(
        population_avec_score_normalise)

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

    # ax.scatter([0], [0], [0], c='g', picker=True, pickradius=5)

   #ax.scatter([s[1] for s in population_avec_score_normalise], [s[2] for s in population_avec_score_normalise],
               #[s[3] for s in population_avec_score_normalise], c='b', picker=True, pickradius=0.1)
    ax.scatter([s[1] for s in pareto_frontier], [s[2] for s in pareto_frontier], [s[3] for s in pareto_frontier], c='r',
               picker=True, pickradius=0.1)
    # ax.legend()
    ax.set_xlabel("Compacité")
    ax.set_xlim(min(liste_compacite), 1000)
    ax.set_ylabel("Proximité")
    ax.set_ylim(min(liste_proximite), max(liste_proximite))
    ax.set_zlabel("Production")
    ax.set_zlim(min(liste_production), max(liste_production))
    fig.canvas.mpl_connect('pick_event', lambda event: onpick(event, pareto_frontier))

    #fig, ax = plt.subplots()
    #ax.plot(best_scores)
    #ax.set_xlabel("Iterations")
    #ax.set_ylabel("Best Final Score")

    plt.show()

    create_2D_projection_image([liste_compacite, liste_proximite, liste_production], [
        pareto_compacite, pareto_proximite, pareto_production])


def onpick(event, normalise):
    ind = event.ind

    print(ind)
    solution = normalise[ind[0]][0]
    print(solution)
    plot_solution(solution)


def create_2D_projection_image(scores, pareto_scores):
    names = ["compacity", "proximity", "production"]
    for i in range(3):
        for j in range(i + 1, 3):
            fig = plt.figure()
            ax2d = fig.add_subplot(111)
            # ax2d.scatter(scores[j], scores[i])
            ax2d.scatter(pareto_scores[j], pareto_scores[i], c="r")
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


""" FITNESS FUNCTIONS """


# Comme tous les critères sont a minimiser, le point idéal est le point (0, 0, 0)

def compacite(solution):
    """ Computes the inverse of mean of the euclidean distance of a bought parcel and center one.
    """
    # Trouver la parcelle du milieu
    milieuX = sum(plot[0] for plot in solution) / len(solution)
    milieuY = sum(plot[1] for plot in solution) / len(solution)
    # Critère à être minimiser
    return sum((plot[0] - milieuX) ** 2 + (plot[1] - milieuY) ** 2 for plot in solution)


def proximite(solution):
    """ Computes the mean of the euclidian distance of a bought parcel and inhabited zone
    """
    # Critère à être minimiser
    return sum(DISTANCE_MAP[solution[i]] for i in range(len(solution))) / len(solution)


def production(solution):
    """ Computes the inverse of the sum of production for each bought parcel
    """
    # Critère à être maximiser mais l'inversion fait qu'on devra la minimiser
    return 1 / sum(PRODUCTION_MAP[solution[i]] for i in range(len(solution)))


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
    # séparer les doublons
    double_solutions = list(set(parent1).intersection(parent2))
    # combiner les parents
    all_possible_parcels = parent1 + parent2
    # enlever les doublons
    unique_parcels = list(set(all_possible_parcels))
    # Generer des enfants
    child1, child2 = [], []
    for i, parcel in enumerate(unique_parcels):
        if i % 2 == 0:  # y avait un +1 qui faisait faussement fonctionner le code
            child1.append(parcel)
        else:
            child2.append(parcel)
    # Si l'enfant est vide, ne pas effectuer la reproduction
    child1.extend(double_solutions), child2.extend(double_solutions)
    if not child1 or not child2:
        print('probleme reprodcution 2')
        return None, None
    while cost_bought_plot(child1) > BUDGET:
        child1.pop()
    while cost_bought_plot(child2) > BUDGET:
        child2.pop()
    while cost_bought_plot(child1) < BUDGET - 80000:
        child1.append((random.randint(0, 69), random.randint(0, 169)))
    while cost_bought_plot(child2) < BUDGET - 80000:
        child2.append((random.randint(0, 69), random.randint(0, 169)))
    return list(set(child1)), list(set(child2))


def reproduction_population(population):
    """ Make the reproduction of a set of solutions

    Args:
        population: set of solutions
    """
    for i in range(0, len(population) - 10, 2):
        # Selection of the parents
        random1 = random.randint(0, len(population) - 1)
        random2 = random.randint(0, len(population) - 1)
        while random1 == random2:
            random2 = random.randint(0, len(population) - 1)

        parent1 = population[random1]
        parent2 = population[random2]
        # Generation of two children
        child1, child2 = reproduction(parent1, parent2)


        if child1 is not None and child2 is not None :
            population.append(child1)
            population.append(child2)
        return population



def mutation_population(population):
    nouvelle_solution_mutee = []
    for solution in population:
        copie_solution = copy.deepcopy(solution)

        new_plot_flat_index = np.random.choice(COST_MAP.size)
        new_plot_index = np.unravel_index(new_plot_flat_index, COST_MAP.shape)
        while USAGE_MAP[new_plot_index] != 0 or new_plot_index in copie_solution:
            new_plot_flat_index = np.random.choice(COST_MAP.size)
            new_plot_index = np.unravel_index(new_plot_flat_index, COST_MAP.shape)

        copie_solution.append(new_plot_index)
        while cost_bought_plot(copie_solution) > BUDGET:
            copie_solution.pop(random.randint(0, len(copie_solution) - 1))
        nouvelle_solution_mutee.append(copie_solution)

    population.extend(nouvelle_solution_mutee)

    return population



def selection(population, population_size):
    population_without_doubles = []
    for solution in population:
        if solution not in population_without_doubles:
            population_without_doubles.append(solution)
    population = population_without_doubles

    pop_avec_norm_score = population_with_normalized_score(population)




    # Supprimer solutions doublons
    #for sol1_ac_score in pop_avec_norm_score:
    #for sol2_ac_score in pop_avec_norm_score:

    """# Check si les deux solutions sont différentes et que les scores sont très proches
    difference_compacite = abs(sol1_ac_score[1] - sol2_ac_score[1])
    difference_proximite = abs(sol1_ac_score[2] - sol2_ac_score[2])
    difference_production = abs(sol1_ac_score[3] - sol2_ac_score[3])
    if sol1_ac_score != sol2_ac_score and difference_compacite < 0.05 \
            and difference_proximite < 0.05 \
            and difference_production < 0.05:
        pop_avec_norm_score.remove(sol1_ac_score)
        break"""
    filtered_population = [solution[0] for solution in pop_avec_norm_score]


    # Ajouter des solutions aléatoire pour avoir la bonne taille
    while len(filtered_population) < population_size:
        filtered_population.append(solution_generator())

    # Réévaluer la population actuelle
    population_avec_score_separe = population_with_normalized_score(filtered_population)
    # Trouver solutions dominantes par Pareto
    population_ac_score_pareto = []
    for solution1 in population_avec_score_separe:
        score_pareto = 0
        for solution2 in population_avec_score_separe:
            score_pareto += dominance(solution1, solution2)
        population_ac_score_pareto.append((solution1[0], score_pareto))
    sorted_pareto_liste = sorted(population_ac_score_pareto, key=lambda x: x[1], reverse=True)

    # Ajouter le meilleur score pour plotter
    best_scores.append(sorted_pareto_liste[0][1])

    # Selectionner les meilleures solutions par dominance de Pareto
    sorted_liste = []
    for solution3 in sorted_pareto_liste:
        sorted_liste.append(solution3[0])
    return sorted_liste[:population_size]


def eliminate_doubles(population_avec_final_score_trié):
    # sort the list based on the float value in each element
    population_avec_final_score_trié.sort(key=lambda x: x[1])
    # iterate over the list and delete elements with the same score
    scores = set()
    for item in population_avec_final_score_trié:
        if item[1] in scores:
            population_avec_final_score_trié.remove(item)
        else:
            scores.add(item[1])


def genetic_algorithm(initial_population_size, iteration):
    """ Genetic Algorithm:
    1: Initial population
    2: Evolution loop
        2.1: Reproduction
        2.2: Mutation
        2.3: Selection
    3: Optimal solution plot
    """
    initial_population = population_generator(initial_population_size)
    nouvelle_population = initial_population[:]
    for _ in tqdm(range(iteration)):
        nouvelle_population = reproduction_population(nouvelle_population)
        nouvelle_population = mutation_population(nouvelle_population)
        nouvelle_population = selection(nouvelle_population, initial_population_size)
    print(" Solution Cost: € {:,}".format(
        cost_bought_plot(nouvelle_population[0])))
    #plot_solution(nouvelle_population[0])
    plot_pareto(nouvelle_population)

    # Tester s'il y a des doublons dans la solution finale
    # find_double_sublists(nouvelle_population)

    return nouvelle_population


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
            (population[i], score_separe(population[i])[0], score_separe(population[i])[1], score_separe(population[i])[2]))
    return generation_avec_score


def population_with_normalized_score(population):
    """ Normalize the criteria scores for each solution in the population
    """
    generation_avec_score = population_with_separate_score(population)

    max_compacite = max(generation_avec_score, key=lambda x: x[1])[1]
    min_compacite = min(generation_avec_score, key=lambda x: x[1])[1]
    max_proximite = max(generation_avec_score, key=lambda x: x[2])[2]
    min_proximite = min(generation_avec_score, key=lambda x: x[2])[2]
    max_production = max(generation_avec_score, key=lambda x: x[3])[3]
    min_production = min(generation_avec_score, key=lambda x: x[3])[3]







    """
        # Trouver l'amplitude maximale pour chaque critère
        max_compacite = max(generation_avec_score, key=lambda x: x[1][0])[1][0]
        min_compacite = min(generation_avec_score, key=lambda x: x[1][0])[1][0]
        max_proximite = max(generation_avec_score, key=lambda x: x[1][1])[1][1]
        min_proximite = min(generation_avec_score, key=lambda x: x[1][1])[1][1]
        max_production = max(generation_avec_score, key=lambda x: x[1][2])[1][2]
        min_production = min(generation_avec_score, key=lambda x: x[1][2])[1][2]"""

    # Normaliser les scores pour chaque critère
    population_with_normalized_score = []
    for i in range(len(generation_avec_score)):
        population_with_normalized_score.append((generation_avec_score[i][0],
                                                 (generation_avec_score[i][1] - min_compacite) / (
                                                         max_compacite - min_compacite),
                                                 (generation_avec_score[i][2] - min_proximite) / (
                                                         max_proximite - min_proximite),
                                                 (generation_avec_score[i][3] - min_production) / (
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
        # Score global à minimiser
        score_global = population_with_score_normalized[i][1] * 0.34 + \
                       population_with_score_normalized[i][2] * 0.33 + \
                       population_with_score_normalized[i][3] * 0.33
        population_with_final_score.append(
            (population_with_score_normalized[i][0], score_global))
    return population_with_final_score


""" Pareto Frontier Functions"""


def dominance(solution1, solution2):
    score_pareto = 0
    if solution1[1] <= solution2[1]:
        score_pareto += 1
    if solution1[2] <= solution2[2]:
        score_pareto += 1
    if solution1[3] <= solution2[3]:
        score_pareto += 1
    return score_pareto


def is_dominant(solution, other_solution):
    """
    Check if a solution is dominant over another solution based on three criteria.

    Returns:
    bool: True if solution is dominant over other_solution, False otherwise.
    """
    if solution[1] <= other_solution[1] or solution[2] <= other_solution[2] or solution[3] <= other_solution[3]:
        return True
    return False


def is_fully_dominant(solution, other_solution):
    """
    Check if a solution is dominant over another solution based on three criteria.

    Returns:
    bool: True if solution is dominant over other_solution, False otherwise.
    """
    if solution[1] <= other_solution[1] and solution[2] <= other_solution[2] and solution[3] <= other_solution[3]:
        return True
    return False


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
            if i != j and is_fully_dominant(other_solution, solution):
                is_pareto_optimal = False
                remaining_solutions.append(solution)
                break
        # Si la solution est optimal, ajouter à la liste
        if is_pareto_optimal:
            pareto_frontier.append(solution)
    return pareto_frontier, remaining_solutions


def find_double_sublists(lst):
    double_sublists = []
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if len(set(lst[i] + lst[j])) == 2:
                double_sublists.append([lst[i], lst[j]])
    print(double_sublists)



def electre(population_finale, poids_compacite = 0.33, poids_proximite = 0.33, poids_production = 0.33):
    population_avec_score_normalise = population_with_normalized_score(population_finale)
    pareto_frontier, population_avec_score_normalise = get_pareto_frontier(
        population_avec_score_normalise)
    liste_de_scores_finaux = []
    for solution in pareto_frontier:
        solution_avec_score_finale = (solution[0], solution[1]*poids_compacite + solution[2]*poids_proximite + solution[3]*poids_production)
        liste_de_scores_finaux.append(solution_avec_score_finale)
    liste_de_scores_finaux_triee = sorted(liste_de_scores_finaux, key=lambda x: x[1], reverse=False)
    return liste_de_scores_finaux_triee[0][0]


if __name__ == "__main__":
    """1: Loading the problem's maps"""
    MAP_DIMENSION = get_map_dimension(COST_MAP_PATH)
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
    population_amelioree = genetic_algorithm(500, 300)
    print(population_amelioree[0])
    # Determine the dominant solutions
    # Plot the frontier and generate csv files

    """3: MCDA: ELECTRE or PROMETHEE"""

    #poids_proximite = int(input("poids proximité (33% si rien):"))
    #poids_compacite = int(input("poids compacité (33% si rien):"))
    #poids_production = 1 - poids_compacite - poids_proximite
    solution_finale = electre(population_amelioree)
    print(solution_finale)
    plot_solution(solution_finale)


    # Rank the solutions from the Pareto Frontier according to ELECTRE/PROMETHEE.
