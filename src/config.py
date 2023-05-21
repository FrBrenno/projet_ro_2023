import random
import numpy as np

""" PARAMETERS  """



# MAP PARAMETERS
COST_MAP_PATH = "Cost_map.txt"
PRODUCTION_MAP_PATH = "Production_map.txt"
USAGE_MAP_PATH = "Usage_map.txt"
MAP_COST_RATIO = 10000
BUDGET = 500000


# Random seed
SEED = 2           #0
random.seed(SEED)
np.random.seed(SEED)

# ALGORITHM PARAMETERS
""" GENETIC PARAMETERS """
POPULATION_SIZE = 400   # 400
NB_ITERATION = 600     # 600

SEUIL_DIFF_COMPACITE = 0.015 # 0.015
SEUIL_DIFF_PROXIMITE = 0.015 # 0.015
SEUIL_DIFF_PRODUCTION = 0.015 # 0.015

""" PROMETHEE PARAMETERS """
WEIGHTS = [0.34, 0.33, 0.33]                # [0.34, 0.33, 0.33]
INDIFFERENCE_THRESHOLD_EMPLACEMENT = 0.2    # 0.4
PREFERENCE_THRESHOLD_EMPLACEMENT = 0.8      # 0.6

# TEST MAP PARAMETERS
TEST_MAP_PATH = "20x20_"
TEST_MAP_DIMENSION = (20, 20)

# CONFIGURATION
USING_TEST_MAP = False
USING_DATA_PLOT = False
USE_EVOLUTION_LOOP = False
PLOTTING_BLUE_POINTS = False




