from src.helper import generate_csv
from src.config import INDIFFERENCE_THRESHOLD_EMPLACEMENT, PREFERENCE_THRESHOLD_EMPLACEMENT, WEIGHTS

""" PROMETHEE FUNCTIONS """


def compute_thresholds(dataset):
    """
    Computes the thresholds for the preference function.
    Indifference threshold is set to 40% of the range width.
    Preference threshold is set to 60% of the range width.
    """
    indiff_threshold = []
    pref_threshold = []
    for i in range(1, len(dataset[0])):
        min_value = min([row[i] for row in dataset])
        max_value = max([row[i] for row in dataset])
        range_width = (max_value - min_value)
        indifference_threshold = range_width * INDIFFERENCE_THRESHOLD_EMPLACEMENT
        preference_threshold = range_width * PREFERENCE_THRESHOLD_EMPLACEMENT

        indiff_threshold.append(indifference_threshold)
        pref_threshold.append(preference_threshold)
    return indiff_threshold, pref_threshold


def preference(score_sol_1, score_sol_2, seuil_indeference, seuil_preference):
    """
    Computes the preference degree of solution_1 over solution_2.
    PARAMETERS: SEUIL_INDEFERENCE, SEUIL_PREFERENCE
    Returns: Preference degree of solution_1 over solution_2.
    """
    # Comme tous les critères sont à minimiser, inversion la différence
    difference = score_sol_2 - score_sol_1

    if difference < seuil_indeference:
        return 0
    elif difference > seuil_preference:
        return 1
    else:
        return (difference - seuil_indeference) / (seuil_preference - seuil_indeference)


def global_preference(solution_1, solution_2, SEUIL_INDIFFERENCE, SEUIL_PREFERENCE):
    """
    Computes the global preference of the solution_1 over solution_2.
    Returns: Preference degree of solution_1 over solution_2.
    """
    preference_list = []
    for i in range(1, len(solution_1)):
        criteria_preference = preference(solution_1[i], solution_2[i], SEUIL_INDIFFERENCE[i - 1],
                                         SEUIL_PREFERENCE[i - 1])
        preference_list.append(criteria_preference)
    return sum([WEIGHTS[j] * preference_list[j] for j in range(len(preference_list))])


def generate_preference_matrix(population_avec_score, SEUIL_INDIFFERENCE, SEUIL_PREFERENCE):
    preference_matrix = []
    for i, solution_1 in enumerate(population_avec_score):
        row = []
        for j, solution_2 in enumerate(population_avec_score):
            if i != j:
                preference_sol1_over_sol2 = round(
                    global_preference(solution_1, solution_2, SEUIL_INDIFFERENCE, SEUIL_PREFERENCE), 5)
                row.append(preference_sol1_over_sol2)
            else:
                row.append(0)
        preference_matrix.append(row)
    return preference_matrix


def compute_flow_scores(preference_matrix):
    net_flow_scores = []
    for i, solution_1 in enumerate(preference_matrix):
        positive_flow_score = 0
        negative_flow_score = 0
        for j, solution_2 in enumerate(preference_matrix):
            if i != j:
                positive_flow_score += preference_matrix[i][j]
                negative_flow_score += preference_matrix[j][i]
        new_flow = round((positive_flow_score - negative_flow_score) / (len(preference_matrix) - 1), 5)
        net_flow_scores.append(new_flow)
    return net_flow_scores


def promethee(population, SEUIL_INDIFFERENCE, SEUIL_PREFERENCE):
    """
    Rank the solutions using the PROMETHEE method.
    # 1: Generate a preference matrix
        For every pair of solutions
            Calculate preference degree
            Calculate the global preference for every pair of solution
    # 2: Computes positive, negative and net flow scores
    # 3: Rank the solutions
    """
    preference_matrix = generate_preference_matrix(population, SEUIL_INDIFFERENCE, SEUIL_PREFERENCE)
    net_flow_scores = compute_flow_scores(preference_matrix)
    ranked_solutions = sorted(zip(population, net_flow_scores), key=lambda x: x[1], reverse=True)

    # Generate csv files with the ranked solutions and its net flow score
    generate_csv("ranked_solutions", ranked_solutions, ["solution, compacity, proximity, production", "net_flow_score"])

    return ranked_solutions
