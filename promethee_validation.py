#####
# PROMETHEE VALIDATION SCRIPT
# THIS PROBLEM WAS SOLVED BY HAND USING THE SAME PARAMETERS AS HERE. THE RESULTS ARE THE SAME.
# PARAMETERS:
#  1. WEIGHTS = [.3, .5, .2]
#  2. SEUIL_INDIFFERENCE = [0.5, 0.5, 750.0]
#  3. SEUIL_PREFERENCE = [1.5, 1.5, 2250.0]
# THE HANDWRITTEN SOLUTION IS IN THE FILE "promethee_handwritten_solution.pdf" #TODO: ADD THE FILE
#####

from pprint import pprint

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
        seuil_indifference = range_width * 0.40
        seuil_preference = range_width * 0.60

        indiff_threshold.append(seuil_indifference)
        pref_threshold.append(seuil_preference)
    return indiff_threshold, pref_threshold


def preference(score_sol_1, score_sol_2, seuil_indeference, seuil_preference, isToMinimise = False):
    """
    Computes the preference degree of solution_1 over solution_2.
    PARAMETERS: SEUIL_INDEFERENCE, SEUIL_PREFERENCE
    Returns: Preference degree of solution_1 over solution_2.
    """
    if isToMinimise:
        difference = score_sol_2 - score_sol_1
    else:
        difference = score_sol_1 - score_sol_2

    if difference < seuil_indeference:
        return 0
    elif difference > seuil_preference:
        return 1
    else:
        return (difference - seuil_indeference) / (seuil_preference - seuil_indeference)


def global_preference(solution_1, solution_2):
    """
    Computes the global preference of the solution_1 over solution_2.
    PARAMETERS: WEIGHTS
    Returns: Preference degree of solution_1 over solution_2.
    """
    preference_list = []
    c1_preference = preference(solution_1[1], solution_2[1], SEUIL_INDIFFERENCE[0], SEUIL_PREFERENCE[0])
    c2_preference = preference(solution_1[2], solution_2[2], SEUIL_INDIFFERENCE[1], SEUIL_PREFERENCE[1])
    c3_preference = preference(solution_1[3], solution_2[3], SEUIL_INDIFFERENCE[2], SEUIL_PREFERENCE[2], True)
    preference_list.append(c1_preference)
    preference_list.append(c2_preference)
    preference_list.append(c3_preference)
    return sum([WEIGHTS[i] * preference_list[i] for i in range(len(preference_list))])


def generate_preference_matrix(population_avec_score):
    preference_matrix = []
    for i, solution_1 in enumerate(population_avec_score):
        row = []
        for j, solution_2 in enumerate(population_avec_score):
            if i != j:
                preference_sol1_over_sol2 = round(global_preference(solution_1, solution_2), 5)
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
        net_flow = round((positive_flow_score - negative_flow_score) / (len(preference_matrix) - 1), 5)
        net_flow_scores.append(net_flow)
    return net_flow_scores


def promethee(population_amelioree):
    """
    Rank the solutions using the PROMETHEE method.
    # 1: Generate a preference matrix
        For every pair of solutions
            Calculate preference degree
            Calculate the global preference for every pair of solution
    # 2: Computes positive, negative and net flow scores
    # 3: Rank the solutions
    """
    preference_matrix = generate_preference_matrix(population_amelioree)
    print("PREFERENCE MATRIX:")
    pprint(preference_matrix)
    net_flow_scores = compute_flow_scores(preference_matrix)
    print("NET FLOW SCORES:")
    pprint(net_flow_scores)
    ranked_solutions = sorted(zip(population_amelioree, net_flow_scores), key=lambda x: x[1], reverse=True)

    return ranked_solutions


if __name__ == "__main__":
    data = [
        ["Honda Civic", 6, 3, 22000],
        ["Toyota Corolla", 8, 4, 19000],
        ["Mazda 3", 7, 5, 21000],
    ]
    WEIGHTS = [.3, .5, .2]
    SEUIL_INDIFFERENCE, SEUIL_PREFERENCE = compute_thresholds(data)
    print("############### PROMETHEE VALIDATION SCRIPT ####################")
    print("DATA:")
    for item in data:
        print(item)
    print("WEIGHTS: \n{}".format(WEIGHTS))
    print("SEUILS D'INDIFFERENCE: \n{}".format(SEUIL_INDIFFERENCE))
    print("SEUILS DE PREFERRENCE: \n{}".format(SEUIL_PREFERENCE))
    print("#################################################################")

    rank = promethee(data)

    print("#################################################################")
    print("RANK:")
    pprint(rank)
