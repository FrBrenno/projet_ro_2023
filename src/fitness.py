""" FITNESS FUNCTIONS """


# Comme tous les critères sont à minimiser, le point idéal est le point (0, 0, 0)

def compacite(solution):
    """ Computes the inverse of mean of the Euclidean distance of a bought parcel and center one.
    """
    # Trouver la parcelle du milieu
    milieu_x = sum(plot[0] for plot in solution) / len(solution)
    milieu_y = sum(plot[1] for plot in solution) / len(solution)
    return (sum((((plot[0] - milieu_x) ** 2 + (plot[1] - milieu_y) ** 2) ** (
            1 / 2) for plot in solution)) / len(solution))


def proximite(solution, DISTANCE_MAP):
    """ Computes the mean of the euclidian distance of a bought parcel and inhabited zone
    """
    return sum(DISTANCE_MAP[solution[i]] for i in range(len(solution))) / len(solution)


def production(solution, PRODUCTION_MAP):
    """ Computes the inverse of the sum of production for each bought parcel
    """
    return 1 / sum(PRODUCTION_MAP[solution[i]] for i in range(len(solution)))
