""" Pareto Frontier Functions"""


def dominance(solution1, solution2):
    score_pareto = 0
    if solution1[1] <= solution2[1]:
        score_pareto += 1
    if solution1[2] <= solution2[2]:
        score_pareto += 1
    if solution1[3] <= solution2[3]:
        score_pareto += 1
    return score_pareto


def is_fully_dominant(solution, other_solution):
    """
    Check if a solution is dominant over another solution based on three criteria.

    Returns:
    bool: True if solution is dominant over other_solution, False otherwise.
    """
    if solution[1] <= other_solution[1] and solution[2] <= other_solution[2] and solution[3] <= other_solution[3]:
        return True
    return False


def get_pareto_frontier(solutions):
    """ Generates a list containing the pareto optimal solutions from solutions
    """
    pareto_frontier = []
    remaining_solutions = []
    # Enumerer chaque solution afin de ne pas comparer des solutions identiques
    for i, solution in enumerate(solutions):
        is_pareto_optimal = True
        for j, other_solution in enumerate(solutions):
            # Si les solutions sont différentes et qu'elle est dominé, alors elle n'est pas optimal
            if i != j and is_fully_dominant(other_solution, solution):
                is_pareto_optimal = False
                remaining_solutions.append(solution)
                break
        # Si la solution est optimal, ajouter à la liste
        if is_pareto_optimal:
            pareto_frontier.append(solution)
    return pareto_frontier, remaining_solutions
