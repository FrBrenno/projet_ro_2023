import copy
import os

import matplotlib.pyplot as plt

from src.genetic_algorithm import cost_bought_plot
from src.fitness import *
from src.config import PLOTTING_BLUE_POINTS


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


def plot_solution(solution, COST_MAP, USAGE_MAP, PRODUCTION_MAP):
    """
    Plot the solution on a map.
    """
    bought_plot = copy.deepcopy(USAGE_MAP)
    for i in range(len(solution)):
        bought_plot[solution[i]] = 5

    fig, axs = plt.subplots(1, 1, figsize=(10, 9))
    fig.canvas.manager.set_window_title(
        "solution Plot: " + "coût: " + str(cost_bought_plot(solution, COST_MAP)) + "  compacity: " + str(
            compacite(solution)) + " proximite: " + str(proximite(solution, COST_MAP)) + " production: " + str(
            1 / production(solution, PRODUCTION_MAP)))
    plt.imshow(bought_plot, cmap='gray', interpolation='nearest')
    fig.savefig(f'./img/best_solution.png')
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
                   [s[3] for s in population_avec_score_normalise], c='b', picker=True, pickradius=0.1)
    #plot the pareto frontier
    ax.scatter([s[1] for s in pareto_frontier], [s[2] for s in pareto_frontier], [s[3] for s in pareto_frontier], c='r',
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


def onpick(event, normalise, COST_MAP, USAGE_MAP, PRODUCTION_MAP):
    """
    When a point is clicked on the pareto frontier, plot the solution corresponding to this point.
    """
    ind = event.ind
    solution = normalise[ind[0]][0]
    plot_solution(solution, COST_MAP, USAGE_MAP, PRODUCTION_MAP)


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
