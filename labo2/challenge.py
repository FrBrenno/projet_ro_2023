import random
import pandas as pd
import time
import matplotlib.pyplot as plt
from math import sqrt, pow

PROBABILITY_MUTATION = 2
NB_GENERATIONS = 10
NB_MACHINES = 10

time_table = []
number_population_table = []
best_score_table = []
generation_table = []
makespan_table = []
tardiness_table = []
best_makespan_table = []
best_total_tardiness_table = []


def plot_configure():
    plot1 = plt.subplot2grid((1, 2), (0, 0))
    plot1.set_xlabel("Tardiness")
    plot1.set_ylabel("Makespan")
    plot2 = plt.subplot2grid((2, 2), (0, 1))
    plot2.set_xlabel("Generation")
    plot2.set_ylabel("Best Score")
    plot3 = plt.subplot2grid((2, 2), (1, 1), colspan=2)
    plot3.set_xlabel("Generation")
    plot3.set_ylabel("Temps")
    return plot1, plot2, plot3


def init_population(nbPieces, nbMachine):
    """Generates the initial population

    Args:
        nbPieces (int): Number of lines in the datasheet
        nbMachine (int): Number of columns corresponding to differents machine

    Returns:
        list(int): Les premiers parents qui définissent l'ordre à laquelle chaque pièce doit
        passer dans chaque machine
    """
    parents = [[i for i in range(nbPieces)] for j in range(nbMachine)]
    for p in parents:
        # Gives more randomness to the shuffle
        random.seed(random.randint(0, 10*nbPieces))
        random.shuffle(p)
    return parents


def compute_score(tupleMakespanTardiness):
    return sqrt(pow(tupleMakespanTardiness[0], 2) + pow(tupleMakespanTardiness[1], 2))


def compute_makespan_tardiness(parent, nbPiece, nbMachine, priority, deadline, makespan):
    """Computes the makespan and the tardiness for a specific parent who defined the order
    in which the pieces are going to be made.

    Args:
        parent (list): List defining the order that each piece should be made
        nbPiece (int): Number of pieces to be made
        nbMachine (int): Number of machines used
        priority (DataFrame): defines the priority of the piece
        deadline (DataFrame): defines the deadline of the piece
        makespan (DataFrame): defines the time each machine m takes to make a piece p

    Returns:
        tuple: Tuple with the makespan and the tardiness
    """
    makespan_matrix = [[0 for i in range(nbMachine)] for j in range(nbPiece)]
    tardiness = 0
    for piece in range(len(parent)):
        for machine in range(nbMachine):
            # First Piece - First Machine
            machine_makespan = makespan.iloc[parent[piece], machine]
            if (piece == 0 and machine == 0):
                makespan_matrix[piece][machine] = machine_makespan
            # First Piece - following machines
            elif (piece == 0):
                makespan_matrix[piece][machine] += makespan_matrix[piece][machine -
                                                                          1] + machine_makespan
            # Following pieces - first machine
                # This piece can only enter the first machine at the timespan of the piece above it
            elif (machine == 0):
                makespan_matrix[piece][machine] += makespan_matrix[piece -
                                                                   1][machine] + machine_makespan
            # This piece enters the machine at the timespan of completion of piece above it in the machine m
            # or the timespan that it comes out of the first machine m-1
            else:
                makespan_matrix[piece][machine] += max(makespan_matrix[piece-1][machine],
                                                       makespan_matrix[piece][machine-1]) + machine_makespan

            # Computing the tardiness
            if ((machine == nbMachine - 1) and (makespan_matrix[piece][-1] > deadline.iloc[parent[piece]])):
                tardiness += (makespan_matrix[piece][-1] -
                              deadline.iloc[parent[piece]]) * priority.iloc[parent[piece]]
    makespan_value = makespan_matrix[-1][-1]
    return makespan_value, tardiness


def sort_best(parents, scores):
    for element in range(len(scores)):
        for index in range(len(scores)):
            if element != index and scores[element] < scores[index]:
                scores[element], scores[index] = scores[index], scores[element]
                parents[element], parents[index] = parents[index], parents[element]


def mutation(children):
    if (random.randint(0, PROBABILITY_MUTATION) == random.randint(0, PROBABILITY_MUTATION)):
        first_mutation_index = random.randint(0, len(children)-1)
        second_mutation_index = random.randint(0, len(children)-1)
        while (first_mutation_index == second_mutation_index):
            second_mutation_index = random.randint(0, len(children)-1)
        children[first_mutation_index], children[second_mutation_index] = children[second_mutation_index], children[first_mutation_index]


def combine(firstParent, secondParent):
    baseA = firstParent[:len(firstParent)//2]
    baseB = firstParent[len(firstParent)//2+1:]
    childA = baseA + [elem for elem in secondParent if elem not in baseA]
    childB = [elem for elem in secondParent if elem not in baseB] + baseB
    return [childA, childB]


def delete_double(listDouble):
    new_list = []
    for item in listDouble:
        if item not in new_list:
            new_list.append(item)
    return new_list


def reproduction(parents):
    children = []
    for first in parents:
        second = parents[random.randint(0, len(parents)-1)]
        while (first == second):
            second = parents[random.randint(0, len(parents)-1)]
        children += combine(first, second)
        for FREQUENCY_MUTATION in range(random.randint(1, len(parents)//2)):
            mutation(children)
    children = delete_double(children)
    return children


def evaluationPopulation(nbMachine, priority, deadline, makespan, nbPiece, population, scores):
    i = 0
    for piece in population:
        makespanValue, tardinessValue = compute_makespan_tardiness(piece, nbPiece, nbMachine,
                                                                   priority, deadline, makespan)
        score = compute_score((makespanValue, tardinessValue))
        scores.append(score)
        # Sauvegarder des points
        if (len(makespan_table) == 0 and len(tardiness_table) == 0) or (makespanValue < makespan_table[-1]) and (tardinessValue < tardiness_table[-1]):
            makespan_table.append(makespanValue)
            tardiness_table.append(tardinessValue)
        i += 1
    # print(i)
    # Cette boucle se repete 103 fois -> nb de parents? normalement la taille est 56


def naturalSelection(currentGeneration):
    return currentGeneration[:len(currentGeneration)//2-1]


if __name__ == "__main__":
    """Defining columns names"""
    colMachine = [f"t{i+1}" for i in range(NB_MACHINES)]
    colName = ["priority", "deadline", ] + colMachine
    """ Reading data"""
    # data = pd.read_csv(r"./instance.csv", delimiter=",", names=colName)
    data = pd.read_csv(r"./instance.csv", delimiter=",", names=colName)
    priority = data["priority"]
    deadline = data["deadline"]
    makespan = data[[f"t{i+1}" for i in range(NB_MACHINES)]]
    nbPiece = len(data)
    # Minus 2 because there is two columns that gives the priority and the deadline
    """Plot Configuration"""
    plot1, plot2, plot3 = plot_configure()

    optimal_solution = []
    """Population Initial"""
    parents = init_population(
        nbPiece, NB_MACHINES)
    """Calcul du score (Fitness)"""
    scores = []
    evaluationPopulation(NB_MACHINES, priority, deadline,
                         makespan, nbPiece, parents, scores)
    sort_best(parents, scores)
    """Nouvelles Générations"""
    currentGen = parents
    for generation in range(NB_GENERATIONS):
        scores = []
        """Reproduction"""
        currentGen += reproduction(parents)
        tic = time.perf_counter()
        evaluationPopulation(NB_MACHINES, priority, deadline,
                             makespan, nbPiece, currentGen, scores)
        toc = time.perf_counter()
        time_table.append(toc-tic)
        sort_best(currentGen, scores)
        # Sauvegarde des points
        # Si meilleur score
        if (len(best_score_table) == 0) or (scores[0] < best_score_table[-1]):
            optimal_solution = currentGen[0]
            best_score_table.append(scores[0])
            generation_table.append(generation)
        number_population_table.append(len(currentGen))
        currentGen = naturalSelection(currentGen)
        print(f"Generation {generation} done!")
        print(f"Temps evaluation: {toc-tic:0.4f} s")
        print(f"Population number: {len(currentGen)}")
    print(f"Optimal Solution: {optimal_solution}")
    plot1.scatter(tardiness_table, makespan_table, c="green")
    plot1.scatter(tardiness_table[-1], makespan_table[-1], c="red")
    plot1.scatter(best_total_tardiness_table, best_makespan_table, c="blue")
    plot2.plot(generation_table, best_score_table, c="red")
    generation_table = [i for i in range(NB_GENERATIONS)]
    plot3.plot(generation_table, time_table, c="blue")
    ax2 = plot3.twinx()
    ax2.set_ylabel("Number of Current Population ")
    ax2.set_facecolor("orange")
    ax2.plot(generation_table, number_population_table, c="orange")
    plt.show()
