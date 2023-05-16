from src.helper import generate_csv

""" PROMETHEE FUNCTIONS """


def compute_thresholds(data):
    """
    Computes the thresholds for the preference function.
    Threshold interval are centered in the mean of the criteria. Each threshold is half the distance between the mean.
    """
    seuil_indiff = []
    seuil_pref = []
    for i in range(1, len(data[0])):
        min_value = min([row[i] for row in data])
        max_value = max([row[i] for row in data])
        range_mean = (max_value - min_value) / 2
        seuil_indifference = range_mean / 2
        seuil_preference = range_mean * 3 / 2

        seuil_indiff.append(seuil_indifference)
        seuil_pref.append(seuil_preference)
    return seuil_indiff, seuil_pref


def preference(score_sol_1, score_sol_2, seuil_indifference, seuil_preference):
    difference = score_sol_1 - score_sol_2
    if difference < seuil_indifference:
        return 0
    elif difference > seuil_preference:
        return 1
    else:
        return (difference - seuil_indifference) / (seuil_preference - seuil_indifference)


def global_preference(solution_1, solution_2, SEUIL_INDIFFERENCE, SEUIL_PREFERENCE, WEIGHTS):
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


def generate_preference_matrix(population_avec_score, SEUIL_INDIFFERENCE, SEUIL_PREFERENCE, WEIGHTS):
    preference_matrix = []
    for i, solution_1 in enumerate(population_avec_score):
        row = []
        for j, solution_2 in enumerate(population_avec_score):
            if i != j:
                preference_sol1_over_sol2 = global_preference(solution_1, solution_2, SEUIL_INDIFFERENCE, SEUIL_PREFERENCE, WEIGHTS)
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
        net_flow_scores.append((positive_flow_score - negative_flow_score) / (len(preference_matrix) - 1))
    return net_flow_scores


def promethee(population, SEUIL_INDIFFERENCE, SEUIL_PREFERENCE, WEIGHTS):
    """
    Rank the solutions using the PROMETHEE method.
    # 1: Generate a preference matrix
        For every pair of solutions
            Calculate preference degree
            Calculate the global preference for every pair of solution
    # 2: Computes positive, negative and net flow scores
    # 3: Rank the solutions
    """
    preference_matrix = generate_preference_matrix(population, SEUIL_INDIFFERENCE, SEUIL_PREFERENCE, WEIGHTS)
    net_flow_scores = compute_flow_scores(preference_matrix)
    ranked_solutions = sorted(zip(population, net_flow_scores), key=lambda x: x[1], reverse=True)

    # Generate csv files with the ranked solutions and its net flow score
    generate_csv("ranked_solutions", ranked_solutions, ["solution, compacity, proximity, production", "net_flow_score"])

    return ranked_solutions
