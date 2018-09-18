import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt

np.set_printoptions(precision=6)


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


def linearRegCostFunction(X, y, theta, lambda_):
    """Compute cost and gradient for regularized linear
    regression with multiple variables

    [J, grad] = linearRegCostFunction(X, y, theta, lambda) computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y.
    :param X:
    :param y:
    :param theta:
    :param lambda_:
    :return: the cost in J and the gradient in grad
    """
    # Initialize some useful values
    m = len(y)  # number of training examples

    J = computeCost(X, y, theta)
    J += lambda_ / (2 * m) * np.matmul(theta[1:], theta[1:])
    grad = 1 / m * np.matmul(X.transpose(), np.matmul(X, theta) - y)
    grad[1:] += lambda_ / m * theta[1:]
    return J, grad


def trainLinearReg(X, y, lambda_):
    """Trains linear regression given a dataset (X, y) and a
    regularization parameter lambda

    [theta] = trainLinearReg (X, y, lambda) trains linear regression using
    the dataset (X, y) and regularization parameter lambda.
    :param X:
    :param y:
    :param lambda_:
    :return: the trained parameters theta.
    """
    # Initialize Theta
    initial_theta = np.zeros(X.shape[1])

    # Create "short hand" for the cost function to be minimized
    costFunction = lambda t: linearRegCostFunction(X, y, t, lambda_)[0]
    gradFunction = lambda t: linearRegCostFunction(X, y, t, lambda_)[1]

    # Now, costFunction is a function that takes in only one argument
    options = {'maxiter': 200, 'disp': True}

    # Minimize using fmincg
    res = opt.minimize(costFunction, initial_theta, method='CG', jac=gradFunction, options=options)
    theta = res.x
    return theta


def learningCurve(X, y, Xval, yval, lambda_):
    """Generates the train and cross validation set errors needed
    to plot a learning curve

    [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda) returns the train and
    cross validation set errors for a learning curve. In particular,
    it returns two vectors of the same length - error_train and
    error_val. Then, error_train(i) contains the training error for
    i examples (and similarly for error_val(i)).
    :param X:
    :param y:
    :param Xval:
    :param yval:
    :param lambda_:
    :return:
    """
    # Number of training examples
    m = X.shape[0]
    error_train = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(m):
        theta = trainLinearReg(X[:i + 1], y[:i + 1], lambda_)
        error_train[i] = linearRegCostFunction(X[:i + 1], y[:i + 1], theta, 0)[0]
        error_val[i] = linearRegCostFunction(Xval, yval, theta, 0)[0]

    return error_train, error_val


def polyFeatures(X, p):
    """Maps X (1D vector) into the p-th power

    [X_poly] = polyFeatures(X, p) takes a data matrix X (size m x 1) and
    maps each example into its polynomial features where
    X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p]
    :param X:
    :param p:
    :return:
    """
    X_poly = np.zeros((X.shape[0], p))
    for i in range(p):
        X_poly[:, i] = X ** (i + 1)
    return X_poly


def featureNormalize(X):
    """Normalizes the features in X

    featureNormalize(X) returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    :param X:
    :return:
    """
    mu = X.mean(axis=0)
    X_norm = X - mu

    sigma = X_norm.std(axis=0)
    X_norm = X_norm / sigma
    return X_norm, mu, sigma


def plotFit(min_x, max_x, mu, sigma, theta, p):
    """Plots a learned polynomial regression fit over an existing figure.
    Also works with linear regression.

    plotFit(min_x, max_x, mu, sigma, theta, p) plots the learned polynomial
    fit with power p and feature normalization (mu, sigma).
    :param min_x:
    :param max_x:
    :param mu:
    :param sigma:
    :param theta:
    :param p:
    :return:
    """
    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.arange(min_x - 15, max_x + 25.05, 0.05)

    # Map the X values
    X_poly = polyFeatures(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly / sigma

    # Add ones
    X_poly = np.insert(X_poly, 0, 1, axis=1)

    # Plot
    plt.plot(x, np.matmul(X_poly, theta), '--', linewidth=2)


def validationCurve(X, y, Xval, yval):
    """Generate the train and validation errors needed to
    plot a validation curve that we can use to select lambda

    [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval) returns the train
    and validation errors (in error_train, error_val)
    for different values of lambda. You are given the training set (X,
    y) and validation set (Xval, yval).
    :param X:
    :param y:
    :param Xval:
    :param yval:
    :return:
    """
    # Selected values of lambda (you should not change this)
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    error_train = []
    error_val = []

    for lambda_ in lambda_vec:
        theta = trainLinearReg(X, y, lambda_)
        error_train.append(linearRegCostFunction(X, y, theta, 0)[0])
        error_val.append(linearRegCostFunction(Xval, yval, theta, 0)[0])

    return lambda_vec, error_train, error_val
