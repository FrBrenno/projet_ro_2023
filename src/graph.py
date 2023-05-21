import copy
import os

import matplotlib.pyplot as plt

from src.genetic_algorithm import cost_bought_plot
from src.fitness import *
from src.config import *


def configure_data_plot(COST_MAP, PRODUCTION_MAP, USAGE_MAP, DISTANCE_MAP):
    """ Configure a figure where the matrix data is plotted.
    """
    # Create a figure with three subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    fig.canvas.manager.set_window_title("Matrix Data Plot")
    # Plot each matrix in a different subplot

    # COST MATRIX
    axs[0][0].set_title("Cost map")
    # Higher the costs are, the more Yellow it is
    axs[0][0].imshow(COST_MAP, cmap='inferno', interpolation='nearest')

    # PRODUCTION MATRIX
    axs[0][1].set_title("Production map")
    # Higher the productivity is, darker the green is
    axs[0][1].imshow(PRODUCTION_MAP, cmap='Greens',
                     interpolation='nearest')

    # USAGE MATRIX
    axs[1][0].set_title("Usage map")
    axs[1][0].imshow(USAGE_MAP, cmap='gray', interpolation='nearest')

    # DISTANCE MATRIX
    axs[1][1].set_title("Distance map")
    # Higher the distance is, darker the parcel is
    axs[1][1].imshow(DISTANCE_MAP, cmap='Blues',
                     interpolation='nearest')


def plot_solution(solution, COST_MAP, USAGE_MAP, PRODUCTION_MAP,  img_name):
    """
    Plot the solution on a map.
    """
    bought_plot = copy.deepcopy(USAGE_MAP)
    for i in range(len(solution)):
        bought_plot[solution[i]] = 5

    fig, axs = plt.subplots(1, 1, figsize=(10, 9))
    fig.canvas.manager.set_window_title(img_name)
    plt.title(img_name +"\n" + "Coût: " + str(cost_bought_plot(solution, COST_MAP)) + " - Compacity: " + str(round(
        compacite(solution), 4)) + " - Proximite: " + str(round(proximite(solution, COST_MAP), 4)) + " - Production: " + str(
        round(1 / production(solution, PRODUCTION_MAP), 4)))
    plt.imshow(bought_plot, cmap='gray', interpolation='nearest')

    fig.savefig(f'./img/{img_name}.png', )
    plt.show()




def plot_pareto(pareto_frontier, population_avec_score_normalise, COST_MAP, PRODUCTION_MAP, USAGE_MAP):
    """
    Plot the pareto frontier on a map.
    Allow to click on a point to see the solution.
    """
    # determine les listes des valeurs de chaque critère
    liste_compacite = [population_avec_score_normalise[i][1]
                       for i in range(len(population_avec_score_normalise))]
    liste_proximite = [population_avec_score_normalise[i][2]
                       for i in range(len(population_avec_score_normalise))]
    liste_production = [population_avec_score_normalise[i][3]
                        for i in range(len(population_avec_score_normalise))]

    pareto_compacite = [pareto_frontier[i][1]
                        for i in range(len(pareto_frontier))]
    pareto_proximite = [pareto_frontier[i][2]
                        for i in range(len(pareto_frontier))]
    pareto_production = [pareto_frontier[i][3]
                         for i in range(len(pareto_frontier))]

    # Plot the pareto frontier
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.canvas.manager.set_window_title("Pareto Graph")

    if PLOTTING_BLUE_POINTS: # Plot the blue points wich are not in the pareto frontier
        ax.scatter([s[1] for s in population_avec_score_normalise], [s[2] for s in population_avec_score_normalise],
                   [s[3] for s in population_avec_score_normalise], c='b', s=7, picker=True, pickradius=0.1)
    #plot the pareto frontier
    ax.scatter([s[1] for s in pareto_frontier], [s[2] for s in pareto_frontier], [s[3] for s in pareto_frontier], c='r', s=11,
               picker=True, pickradius=0.1)

    # define the labels
    ax.set_xlabel("Compacité")
    ax.set_xlim(min(liste_compacite), max(liste_compacite))
    ax.set_ylabel("Proximité")
    ax.set_ylim(min(liste_proximite), max(liste_proximite))
    ax.set_zlabel("Production")
    ax.set_zlim(min(liste_production), max(liste_production))
    fig.canvas.mpl_connect('pick_event',
                           lambda event: onpick(event, pareto_frontier, COST_MAP, USAGE_MAP, PRODUCTION_MAP))

    plt.show()

    create_2D_projection_image([liste_compacite, liste_proximite, liste_production], [
        pareto_compacite, pareto_proximite, pareto_production])

    save_config()


def onpick(event, normalise, COST_MAP, USAGE_MAP, PRODUCTION_MAP):
    """
    When a point is clicked on the pareto frontier, plot the solution corresponding to this point.
    """
    ind = event.ind
    solution = normalise[ind[0]][0]
    plot_solution(solution, COST_MAP, USAGE_MAP, PRODUCTION_MAP, f"solution_{ind[0]}")


def create_2D_projection_image(scores, pareto_scores):
    """
    Create 2D projection images.
    """
    names = ["compacity", "proximity", "production"]
    for i in range(3):
        for j in range(i + 1, 3):
            fig = plt.figure()
            ax2d = fig.add_subplot(111)
            ax2d.scatter(scores[j], scores[i])
            ax2d.scatter(pareto_scores[j], pareto_scores[i], c="r")
            ax2d.set_xlabel(f'{names[j]}')
            ax2d.set_ylabel(f'{names[i]}')
            os.makedirs('./img', exist_ok=True)
            fig.savefig(f'./img/2Dprojection_{names[j]}_{names[i]}.png')
            plt.close(fig)


def save_config():
    """
    Écris les paramètres de l'algo dans un fichier txt dans le dossier results
    """
    os.makedirs('./results', exist_ok=True)
    with open("./results/config.txt", "w") as f:
        # MAP PARAMETERS
        f.write("MAP PARAMETERS \n")
        f.write(f"MAP_COST_RATIO = {MAP_COST_RATIO}\n")
        f.write(f"BUDGET = {BUDGET}\n")
        f.write("\n \n")

        # ALGORITHM PARAMETERS
        f.write("ALGORITHM PARAMETERS \n")
        f.write(f"SEED = {SEED}\n")
        f.write(f"POPULATION_SIZE = {POPULATION_SIZE}\n")
        f.write(f"NB_ITERATION = {NB_ITERATION}\n")
        f.write("\n")
        f.write(f"SEUIL_DIFF_COMPACITE = {SEUIL_DIFF_COMPACITE}\n")
        f.write(f"SEUIL_DIFF_PROXIMITE = {SEUIL_DIFF_PROXIMITE}\n")
        f.write(f"SEUIL_DIFF_PRODUCTION = {SEUIL_DIFF_PRODUCTION}\n")
        f.write("\n \n")


        # PROMETHEE PARAMETERS
        f.write("PROMETHEE PARAMETERS \n")
        f.write(f"WEIGHTS = {WEIGHTS}\n")
        f.write(f"INDIFFERENCE_THRESHOLD_EMPLACEMENT = {INDIFFERENCE_THRESHOLD_EMPLACEMENT}\n")
        f.write(f"PREFERENCE_THRESHOLD_EMPLACEMENT = {PREFERENCE_THRESHOLD_EMPLACEMENT}\n")





