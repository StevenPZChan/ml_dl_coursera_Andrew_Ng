import numpy as np
import matplotlib.pyplot as plt


def warmUpExercise():
    """Example function

    A = warmUpExercise() is an example function that returns the 5x5 identity matrix
    :return: the 5x5 identity matrix
    """
    return np.identity(5)


def plotData(x, y):
    """Plots the data points x and y into a new figure

    plotData(x,y) plots the data points and gives the figure axes labels of population and profit.
    :param x: population data
    :param y: profit data
    :return:
    """
    figure = plt.figure(figsize=(10, 8))
    plt.plot(x, y, 'rx', markersize=10)
    plt.xlabel('population (in 10,000)')
    plt.ylabel('profit (in $10,000)')
    return figure
