import numpy as np

# MAP PARAMETERS
MAP_DIMENSION=70
COST_MAP_PATH="Cost_map.txt"
PRODUCTION_MAP_PATH="Production_map.txt"
USAGE_MAP_PATH="Usage_map.txt"

# TEST MAP PARAMETERS
TEST_MAP_PATH="20x20_"
TEST_MAP_DIMENSION=20

# CONFIGURATION
USING_TEST_MAP=True
def load_usage_map(file_path):
    map_dimension = TEST_MAP_DIMENSION if USING_TEST_MAP else MAP_DIMENSION
    path = "./donnees/" + TEST_MAP_PATH + file_path if USING_TEST_MAP else "./donnees/" + file_path
    # Create an empty matrix
    matrix = np.zeros((map_dimension, map_dimension), dtype=int)
    # Load the file
    with open(path) as file:
        line_nb = 0
        for line in file.readlines():
            for element_nb in range(len(line)-1):
                element = line[element_nb]
                if element == '' or element == ' ':
                    # Parcelle Inoccupée
                    matrix[line_nb,element_nb] = 0
                elif element == '\n':
                    break
                elif element == 'R':
                    # Parcelle occupée par une route
                    matrix[line_nb,element_nb] = 1
                elif element == 'C':
                    # Parcelle habitée
                    matrix[line_nb,element_nb] = 2
            line_nb += 1
    return matrix
    
def load_map(file_path):
    map_dimension = TEST_MAP_DIMENSION if USING_TEST_MAP else MAP_DIMENSION
    path = "./donnees/" + TEST_MAP_PATH + file_path if USING_TEST_MAP else "./donnees/" + file_path
    # Load the file
    arr = np.loadtxt(path, dtype='str')
    # Create an empty matrix
    matrix = np.zeros((map_dimension, map_dimension), dtype=int)
    # Loop over each row and character to populate the matrix
    for i in range(map_dimension):
        for j in range(map_dimension):
            matrix[i,j] = int(arr[i][j])
    return matrix


if __name__ == "__main__":
    """Loading the problem's maps"""
    cost_map = load_map(COST_MAP_PATH)
    production_map = load_map(PRODUCTION_MAP_PATH)
    usage_map = load_usage_map(USAGE_MAP_PATH)
    print("Cost map:\n", cost_map)
    print("Production map:\n", production_map)
    print("Usage map:\n", usage_map)