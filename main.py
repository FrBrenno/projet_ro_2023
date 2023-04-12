import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import random

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


def matrice_dist(usage_matrice):
    idx_habitations = np.argwhere(usage_matrice == 2)
    # trouver les distances euclidiennes entre chaque parcelle et les parcelles avec une valeur de 2 dans la matrice
    distances = np.min(cdist(np.argwhere(usage_matrice != 2), idx_habitations), axis=1)

    # reconstruire la matrice avec les distances
    distances_mat = np.zeros_like(usage_matrice, dtype=float)
    distances_mat[usage_matrice != 2] = distances
    return distances_mat




def generation_generator(cost_map, usage_map):
    #initialise une matrice du territoire avec que des 0
    bought_plot = np.zeros(MAP_DIMENSION)
    #vérifie le buget
    while np.sum(bought_plot)*MAP_COST_RATIO <= BUDGET:
        # obtenir un index aléatoire dans le tableau aplati
        idx_flat = np.random.choice(cost_map.size)
        # convertir l'index aplati en indices de ligne et de colonne
        idx = np.unravel_index(idx_flat, cost_map.shape)
        #vérifie que c'est pas une route ni une habitation et qu'on a pas déja acheté la parcelle
        if usage_map[idx] == 0 and bought_plot[idx] == 0:
            bought_plot[idx] = cost_map[idx]
            if np.sum(bought_plot)*MAP_COST_RATIO > BUDGET:
                bought_plot[idx] = 0
                break
    return bought_plot


if __name__ == "__main__":
    """1: Loading the problem's maps"""
    cost_map = load_map(COST_MAP_PATH)
    production_map = load_map(PRODUCTION_MAP_PATH)
    usage_map = load_usage_map(USAGE_MAP_PATH)
    distance_map = matrice_dist(usage_map)

    generation_generator(cost_map, usage_map)
    # Plot the matrix data
    if USING_PLOT:
        configure_plot(cost_map, production_map, usage_map, distance_map)
        plt.show()

    """2: Finding the Pareto-Optimal Frontier"""
