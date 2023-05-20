import csv
import os

from scipy.spatial.distance import cdist

from src.config import *

import numpy as np


def get_map_dimension(file_path):
    path = "./data/" + TEST_MAP_PATH + \
           file_path if USING_TEST_MAP else "./data/" + file_path
    with open(path) as file:
        line_nb = 0
        column_nb = 0
        for line in file.readlines():
            line_nb += 1
            if line_nb == 1:
                for element_nb in range(len(line) - 1):
                    column_nb += 1
    return line_nb, column_nb


def load_map(file_path, MAP_DIMENSION):
    """Load maps into memory

    Args:
        MAP_DIMENSION: tuple of the map's dimension
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


def load_usage_map(file_path, MAP_DIMENSION):
    """ Load the usage map into memory

    Args:
        MAP_DIMENSION: tuple of the map's dimension
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


def matrice_dist(MAP_DIMENSION, USAGE_MAP):
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


def generate_csv(filename, data, column_names):
    os.makedirs('./results', exist_ok=True)
    with open('./results/' + filename + '.csv', 'w+', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)
        for row in data:
            writer.writerow(row)
