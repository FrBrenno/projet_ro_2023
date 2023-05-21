from src.helper import *
from src.graph import *
from src.genetic_algorithm import genetic_algorithm
from src.score import population_with_separate_score
from src.promethee import compute_thresholds, promethee
from src.pareto import get_pareto_frontier




if __name__ == "__main__":
    save_config();
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

    rank_solution_1 = ranked_solutions[0][0][0]
    plot_solution(rank_solution_1, COST_MAP, USAGE_MAP, PRODUCTION_MAP, "best_solution_promethee")
    generate_csv("best_solution_promethee", rank_solution_1, ["x", "y"])

    rank_solution_2 = ranked_solutions[1][0][0]
    plot_solution(rank_solution_2, COST_MAP, USAGE_MAP, PRODUCTION_MAP, "rank_solution_promethee_2")
    generate_csv("rank_solution_2", rank_solution_2, ["x", "y"])

    rank_solution_3 = ranked_solutions[2][0][0]
    plot_solution(rank_solution_3, COST_MAP, USAGE_MAP, PRODUCTION_MAP, "rank_solution_promethee_3")
    generate_csv("rank_solution_3", rank_solution_3, ["x", "y"])

    worst_solution = ranked_solutions[-1][0][0]
    plot_solution(worst_solution, COST_MAP, USAGE_MAP, PRODUCTION_MAP, "worst_solution_promethee")
    generate_csv("worst_solution", worst_solution, ["x", "y"])
