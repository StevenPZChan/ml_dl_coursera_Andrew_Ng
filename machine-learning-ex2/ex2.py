import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(precision=6)

def plotData(X, y):
    """Plots the data points X and y into a new figure

    plotData(x,y) plots the data points with + for the positive examples
    and o for the negative examples. X is assumed to be a Mx2 matrix.
    :param X: the exam scores
    :param y: the label
    :return:
    """
    positive = y == 1
    negative = y == 0
    plt.figure()
    plt.plot(X[positive, 0], X[positive, 1], 'k+', markeredgewidth=2, markersize=7)
    plt.plot(X[negative, 0], X[negative, 1], 'ko', markerfacecolor='y', markersize=7)


def sigmoid(z):
    """Compute sigmoid function

    g = sigmoid(z) computes the sigmoid of z.
    :param z:
    :return:
    """
    g = np.frompyfunc(lambda x: 1 / (1 + np.exp(-x)), 1, 1)
    return g(z).astype(z.dtype)

def costFunction(theta, X, y):
    """Compute cost and gradient for logistic regression

    J = costFunction(theta, X, y) computes the cost of using theta as the
    parameter for logistic regression and the gradient of the cost
    w.r.t. to the parameters.
    :param theta:
    :param X:
    :param y:
    :return:
    """
    # Initialize some useful values
    m = len(y) # number of training examples

    h_theta = sigmoid(np.matmul(X, theta))
    ones = np.ones(m)
    J = -1 / m * (np.matmul(y, np.log(h_theta)) + np.matmul(ones - y, np.log(ones - h_theta)))
    grad = 1 / m * np.matmul(X.transpose(), h_theta - y)
    return J, grad



def mapFeature(X1, X2):
    """Feature mapping function to polynomial features

    mapFeature(X1, X2) maps the two input features
    to quadratic features used in the regularization exercise.
    :param X1:
    :param X2:
    :return: a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    """
    degree = 6
    out = np.ones(X1.shape[0])
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out = np.column_stack([out, X1 ** (i - j) * X2 ** j])
    return out



def plotDecisionBoundary(theta, X, y):
    """Plots the data points X and y into a new figure with
    the decision boundary defined by theta

    plotDecisionBoundary(theta, X,y) plots the data points with + for the
    positive examples and o for the negative examples.
    :param theta:
    :param X: either
    1) Mx3 matrix, where the first column is an all-ones column for the
       intercept.
    2) MxN, N>3 matrix, where the first column is all-ones
    :param y:
    :return:
    """
    # Plot Data
    plotData(X[:, 1:], y)
    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([min(X[:, 1])-2, max(X[:, 1])+2])

        # Calculate the decision boundary line
        plot_y = -1 / theta[2] * (theta[1] * plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)

        # Legend, specific for the exercise
        plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'], loc='upper right')
        plt.xlim([30, 100])
        plt.ylim([30, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)))
        # Evaluate z = theta * x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = np.matmul(mapFeature(u[i:i + 1], v[j:j + 1]), theta)

        u, v = np.meshgrid(u, v)
        #
        # Plot z = 0
        # Notice you need to specify the range[0, 0]
        cs = plt.contour(u, v, z.transpose(), [0], linewidth=2, colors='green')
        cs.collections[0].set_label('')

def predict(theta, X):
    """Predict whether the label is 0 or 1 using learned logistic
    regression parameters theta

    p = predict(theta, X) computes the predictions for X using a
    threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
    :param theta:
    :param X:
    :return:
    """
    p = sigmoid(np.matmul(X, theta)) >= 0.5
    return p.astype(np.int)

def costFunctionReg(theta, X, y, _lambda):
    """Compute cost and gradient for logistic regression with regularization

    J = costFunctionReg(theta, X, y, lambda) computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    :param theta:
    :param X:
    :param y:
    :param _lambda:
    :return:
    """
    # Initialize some useful values
    m = len(y) # number of training examples
    J, grad = costFunction(theta, X, y)
    J += _lambda / (2 * m) * np.matmul(theta[1:], theta[1:])
    grad[1:] += _lambda / m * theta[1:]
    return J, grad