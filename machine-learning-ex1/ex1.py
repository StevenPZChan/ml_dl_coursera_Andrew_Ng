import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(precision=6)


def warmUpExercise():
    """Example function

    A = warmUpExercise() is an example function that returns the 5x5 identity matrix
    :return: the 5x5 identity matrix
    """
    return np.identity(5)


def plotData(x, y):
    """Plots the data points x and y into a new figure

    plotData(x,y) plots the data points and gives the figure axes labels of
    population and profit.
    :param x: population data
    :param y: profit data
    :return:
    """
    plt.figure()
    plt.plot(x, y, 'rx', markersize=10)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')


def computeCost(X, y, theta):
    """Compute cost for linear regression

    J = computeCost(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
    :param X:
    :param y:
    :param theta:
    :return:
    """
    # Initialize some useful values
    m = len(y)  # number of training examples
    diff = np.matmul(X, theta) - y
    J = 1 / (2 * m) * np.matmul(diff, diff)
    return J


def gradientDescent(X, y, theta, alpha, num_iters):
    """Performs gradient descent to learn theta

    theta = gradientDescent(X, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha
    :param X:
    :param y:
    :param theta:
    :param alpha:
    :param num_iters:
    :return:
    """
    # Initialize some useful values
    m = len(y)  # number of training examples
    J_history = []

    for i in range(num_iters):
        theta -= alpha / m * np.matmul(X.transpose(), np.matmul(X, theta) - y)
        # Save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))

    return theta, J_history
