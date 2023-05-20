from src.helper import *
from src.graph import *
from src.genetic_algorithm import genetic_algorithm
from src.score import population_with_separate_score
from src.promethee import compute_thresholds, promethee
from src.pareto import get_pareto_frontier




if __name__ == "__main__":
    """1: Loading the problem's maps"""
    MAP_DIMENSION = get_map_dimension(COST_MAP_PATH)
    COST_MAP = load_map(COST_MAP_PATH, MAP_DIMENSION)
    PRODUCTION_MAP = load_map(PRODUCTION_MAP_PATH, MAP_DIMENSION)
    USAGE_MAP = load_usage_map(USAGE_MAP_PATH, MAP_DIMENSION)
    DISTANCE_MAP = matrice_dist(MAP_DIMENSION, USAGE_MAP)

    # Plot the matrix data
    if USING_DATA_PLOT:
        configure_data_plot(COST_MAP, PRODUCTION_MAP, USAGE_MAP, DISTANCE_MAP)
        plt.show()

    """2: GENETIC ALGORITHM """

    # Generate initial population randomly ⇾ cover as much as possible the solution space
    population = genetic_algorithm(POPULATION_SIZE, NB_ITERATION, COST_MAP, DISTANCE_MAP, USAGE_MAP, PRODUCTION_MAP)

    population_avec_scores = population_with_separate_score(population, DISTANCE_MAP, PRODUCTION_MAP)
    population_pareto, population_avec_scores = get_pareto_frontier(population_avec_scores)

    plot_pareto(population_pareto, population_avec_scores, COST_MAP, PRODUCTION_MAP, USAGE_MAP)

    # Generate csv files for pareto-optimal solutions
    generate_csv('solutions_pareto', population_avec_scores,
                 ["Solution", "Compacite", "Proximite", "Production"])

    """3: MCDA: PROMETHEE METHOD """
    SEUIL_INDIFFERENCE, SEUIL_PREFERENCE = compute_thresholds(population_pareto)
    ranked_solutions = promethee(population_pareto, SEUIL_INDIFFERENCE, SEUIL_PREFERENCE)

    best_solution = ranked_solutions[0][0][0]
    plot_solution(best_solution, COST_MAP, USAGE_MAP, PRODUCTION_MAP)
    generate_csv("best_solution_promethee", best_solution, ["x", "y"])