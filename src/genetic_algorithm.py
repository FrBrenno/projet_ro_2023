import copy
import random

from src.config import *
from src.score import *
from src.pareto import *

import numpy as np

from tqdm import tqdm



""" UTILS FUNCTIONS """


def cost_bought_plot(solution, COST_MAP):
    """
    Return the cost of a solution
    """
    return sum(COST_MAP[solution[i]] for i in range(len(solution) - 1)) * MAP_COST_RATIO


def solution_generator(COST_MAP, USAGE_MAP):
    """ Generates a random solution to the problem respecting budget and usage constraint
    """
    bought_plot = []
    # vérifie le budget
    while cost_bought_plot(bought_plot, COST_MAP) < BUDGET:
        # obtenir un index aléatoire dans le tableau aplatit
        new_plot_flat_index = np.random.choice(COST_MAP.size)
        # convertir l'index aplati en indices de ligne et de colonne
        new_plot_index = np.unravel_index(new_plot_flat_index, COST_MAP.shape)
        # vérifie que ce n'est pas une route ni une habitation et qu'on n'a pas déjà acheté la parcelle
        if USAGE_MAP[new_plot_index] == 0 and new_plot_index not in bought_plot:
            # ajoute l'index de la parcelle à la liste des parcelles achetées
            bought_plot.append(new_plot_index)
            # vérifie que le budget n'est pas dépassé
            if cost_bought_plot(bought_plot, COST_MAP) > BUDGET:
                # enlève la parcelle dernière parcelle ajoutée de la liste des parcelles achetées
                bought_plot.pop()
                break
    return bought_plot


def population_generator(population_size, COST_MAP, USAGE_MAP):
    """ Generates a set of new solutions
    """
    generation = []
    for i in range(population_size):
        generation.append(solution_generator(COST_MAP, USAGE_MAP))

    return generation


def suppression_double(population):
    """
    Supprime les solutions en double dans la population
    """
    population_without_doubles = []
    for solution in population:
        if solution not in population_without_doubles:
            population_without_doubles.append(solution)
    population = population_without_doubles
    return population


def suppression_sol_trop_proche(population, DISTANCE_MAP, PRODUCTION_MAP):
    """
    Supprime les solutions trop proches dans la population
    """
    pop_avec_norm_score = population_with_normalized_score(population, DISTANCE_MAP, PRODUCTION_MAP)
    pop_avec_norm_score2 = []
    for sol1_ac_score in pop_avec_norm_score:
        is_unique = True
        for sol2_ac_score in pop_avec_norm_score:
            # Check si les deux solutions sont différentes et que les scores sont très proches
            difference_compacite = abs(sol1_ac_score[1] - sol2_ac_score[1])
            difference_proximite = abs(sol1_ac_score[2] - sol2_ac_score[2])
            difference_production = abs(sol1_ac_score[3] - sol2_ac_score[3])
            if sol1_ac_score != sol2_ac_score and difference_compacite < SEUIL_DIFF_COMPACITE and difference_proximite < SEUIL_DIFF_PROXIMITE and difference_production < SEUIL_DIFF_PRODUCTION:
                is_unique = False

        if is_unique:
            pop_avec_norm_score2.append(sol1_ac_score[0])
    return pop_avec_norm_score2


""" GENETIC FUNCTIONS """


def reproduction(parent1, parent2, COST_MAP):
    """ Combine two solutions in order to create to other child solutions
    Args:
        COST_MAP: Cost map of the problem
        parent1: First solution
        parent2: Second solution
    """
    # séparer les doublons
    double_solutions = list(set(parent1).intersection(parent2))
    # combiner les parents et enlever doublons
    all_possible_parcels = parent1 + parent2
    unique_parcels = list(set(all_possible_parcels))
    # Generer des enfants
    child1, child2 = [], []
    for i, parcel in enumerate(unique_parcels):
        if i % 2 == 0:
            child1.append(parcel)
        else:
            child2.append(parcel)
    # Si l'enfant est vide, ne pas effectuer la reproduction
    child1.extend(double_solutions), child2.extend(double_solutions)
    while cost_bought_plot(child1, COST_MAP) > BUDGET:
        child1.pop()
    while cost_bought_plot(child2, COST_MAP) > BUDGET:
        child2.pop()
    return list(set(child1)), list(set(child2))


def reproduction_population(population, COST_MAP):
    """ Make the reproduction of a set of solutions

    Args:
        COST_MAP: Cost map of the problem
        population: set of solutions
    """
    for i in range(0, len(population), 2):
        # Selection of the parents
        random1 = random.randint(0, len(population) - 1)
        random2 = random.randint(0, len(population) - 1)
        while random1 == random2:
            random2 = random.randint(0, len(population) - 1)
        parent1 = population[random1]
        parent2 = population[random2]

        # Generation of two children
        child1, child2 = reproduction(parent1, parent2, COST_MAP)
        if child1 is not None and child2 is not None:
            population.append(child1)
            population.append(child2)
        return population


def mutation_population(population, COST_MAP, USAGE_MAP):
    """ Make the mutation of a set of solutions by adding a new parcel to a solution """

    nouvelle_solution_mutee = []

    for solution in population:
        # Prends une parcelle aléatoire dans l'ensemble des parcelles possibles
        copie_solution = copy.deepcopy(solution)
        new_plot_flat_index = np.random.choice(COST_MAP.size)
        new_plot_index = np.unravel_index(new_plot_flat_index, COST_MAP.shape)

        # Vérifie que la parcelle n'est pas une route ni une habitation et qu'on n'a pas deja acheté la parcelle
        while USAGE_MAP[new_plot_index] != 0 or new_plot_index in copie_solution:
            new_plot_flat_index = np.random.choice(COST_MAP.size)
            new_plot_index = np.unravel_index(new_plot_flat_index, COST_MAP.shape)
        copie_solution.append(new_plot_index)

        # Vérifie que le budget n'est pas dépassé
        while cost_bought_plot(copie_solution, COST_MAP) > BUDGET:
            copie_solution.pop(random.randint(0, len(copie_solution) - 1))
        # Ajoute la solution mutée à la liste des solutions mutées
        nouvelle_solution_mutee.append(copie_solution)

    population.extend(nouvelle_solution_mutee)
    return population


def selection(population, population_size, COST_MAP, DISTANCE_MAP, PRODUCTION_MAP, USAGE_MAP):
    """ Select the best solutions of a set of solutions according to their score """

    # Éliminer les solutions doublons ou trop proches entre elles
    population = suppression_double(population)
    population = suppression_sol_trop_proche(population, DISTANCE_MAP, PRODUCTION_MAP)

    # Ajouter des solutions aléatoires pour avoir la bonne taille de population
    while len(population) < population_size:
        population.append(solution_generator(COST_MAP, USAGE_MAP))

    # Réévaluer la population actuelle
    population_avec_score_separe = population_with_normalized_score(population, DISTANCE_MAP, PRODUCTION_MAP)
    # Trouver solutions dominantes par Pareto
    population_ac_score_pareto = []
    for solution1 in population_avec_score_separe:
        score_pareto = 0
        for solution2 in population_avec_score_separe:
            score_pareto += dominance(solution1, solution2)
        population_ac_score_pareto.append((solution1[0], score_pareto))
    # Trier les solutions par score de dominance de Pareto
    sorted_pareto_liste = sorted(population_ac_score_pareto, key=lambda x: x[1], reverse=True)

    # Selectionner les meilleures solutions par dominance de Pareto
    sorted_liste = []
    for solution3 in sorted_pareto_liste:
        sorted_liste.append(solution3[0])
    return sorted_liste[:population_size]


def genetic_algorithm(initial_population_size, iteration, COST_MAP, DISTANCE_MAP, USAGE_MAP, PRODUCTION_MAP):
    """ Genetic Algorithm:
    1: Initial population
    2: Evolution loop
        2.1: Reproduction
        2.2: Mutation
        2.3: Selection
    3: Optimal solution plot
    """
    initial_population = population_generator(initial_population_size, COST_MAP, USAGE_MAP)
    nouvelle_population = initial_population[:]
    for _ in tqdm(range(iteration)):
        nouvelle_population = reproduction_population(nouvelle_population, COST_MAP)
        nouvelle_population = mutation_population(nouvelle_population, COST_MAP, USAGE_MAP)
        nouvelle_population = selection(nouvelle_population, initial_population_size, COST_MAP, DISTANCE_MAP, PRODUCTION_MAP, USAGE_MAP)

    return nouvelle_population
