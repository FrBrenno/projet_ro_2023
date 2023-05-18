from src.fitness import *


def score_separe(solution, DISTANCE_MAP, PRODUCTION_MAP):
    """ Computes separately each criterion for a given solution
    """
    return compacite(solution), proximite(solution, DISTANCE_MAP), production(solution, PRODUCTION_MAP)


def population_with_separate_score(population, DISTANCE_MAP, PRODUCTION_MAP):
    """ Creates a list of elements containing a solution and each respectively criteria score.
    """
    generation_avec_score = []
    for i in range(len(population)):
        scores_separes = score_separe(population[i], DISTANCE_MAP, PRODUCTION_MAP)
        generation_avec_score.append(
            (population[i], scores_separes[0], scores_separes[1],
             scores_separes[2]))
    return generation_avec_score


def population_with_normalized_score(population, DISTANCE_MAP, PRODUCTION_MAP):
    """ Normalize the criteria scores for each solution in the population
    """
    generation_avec_score = population_with_separate_score(population, DISTANCE_MAP, PRODUCTION_MAP)

    max_compacite = max(generation_avec_score, key=lambda x: x[1])[1]
    min_compacite = min(generation_avec_score, key=lambda x: x[1])[1]
    max_proximite = max(generation_avec_score, key=lambda x: x[2])[2]
    min_proximite = min(generation_avec_score, key=lambda x: x[2])[2]
    max_production = max(generation_avec_score, key=lambda x: x[3])[3]
    min_production = min(generation_avec_score, key=lambda x: x[3])[3]

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
