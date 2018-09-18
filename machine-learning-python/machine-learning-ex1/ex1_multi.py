import numpy as np
import pandas as pd

np.set_printoptions(precision=6)


def featureNormalize(X):
    """Normalizes the features in X

    featureNormalize(X) returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    :param X:
    :return:
    """
    X_norm = pd.DataFrame()
    mu = []
    sigma = []
    for i in range(X.shape[1]):
        temp = np.array(X.iloc[:, i])
        mu.append(np.mean(temp))
        sigma.append(np.std(temp))
        X_norm[i] = (temp - mu[i]) / sigma[i]

    return X_norm, mu, sigma


def computeCostMulti(X, y, theta):
    """Compute cost for linear regression with multiple variables

    J = computeCostMulti(X, y, theta) computes the cost of using theta as the
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


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """Performs gradient descent to learn theta

    theta = gradientDescentMulti(x, y, theta, alpha, num_iters) updates theta by
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
        J_history.append(computeCostMulti(X, y, theta))

    return theta, J_history


def normalEqn(X, y):
    """Computes the closed-form solution to linear regression

    normalEqn(X,y) computes the closed-form solution to linear
    regression using the normal equations.
    :param X:
    :param y:
    :return:
    """
    theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose()), y)
    return theta
