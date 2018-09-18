import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(precision=6)


def plotData(X, y):
    """Plots the data points X and y into a new figure

    plotData(x,y) plots the data points with + for the positive examples
    and o for the negative examples. X is assumed to be a Mx2 matrix.
    :param X:
    :param y:
    :return:
    """
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0
    # Plot Examples
    plt.figure()
    plt.plot(X[pos, 0], X[pos, 1], 'k+', markeredgewidth=1, markersize=7)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', markerfacecolor='y', markersize=7)


def linearKernel(x1, x2):
    """

    linearKernel returns a linear kernel between x1 and x2
    :param x1:
    :param x2:
    :return:
    """
    # Ensure that x1 and x2 are column vectors
    # x1 = x1.reshape(-1)
    # x2 = x2.reshape(-1)

    # Compute the kernel
    sim = np.matmul(x1, x2.transpose())  # dot product
    return sim


def visualizeBoundaryLinear(X, y, model):
    """plots a linear decision boundary learned by the SVM

    visualizeBoundaryLinear(X, y, model) plots a linear decision boundary
    learned by the SVM and overlays the data on it
    :param X:
    :param y:
    :param model:
    :return:
    """
    xp = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    yp = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    xp, yp = np.meshgrid(xp, yp)
    P = model.decision_function(np.column_stack([xp.reshape(-1), yp.reshape(-1)])).reshape(xp.shape)
    plotData(X, y)
    plt.contour(xp, yp, P, '-b', levels=[0])
    plt.show()


def gaussianKernel(x1, x2, sigma):
    """returns a radial basis function kernel between x1 and x2

    sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
    and returns the value in sim
    :param x1:
    :param x2:
    :param sigma:
    :return:
    """
    m = x1.shape[0]
    n = x2.shape[0]
    sim = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            d = x1[i] - x2[j]
            sim[i, j] = np.exp(-np.matmul(d, d) / (2 * sigma ** 2))

    return sim


def visualizeBoundary(X, y, model, *args):
    """plots a non-linear decision boundary learned by the SVM

    visualizeBoundary(X, y, model) plots a non-linear decision
    boundary learned by the SVM and overlays the data on it
    :param X:
    :param y:
    :param model:
    :return:
    """
    # Plot the training data on top of the boundary
    plotData(X, y)

    # Make classification predictions over a grid of values
    x1plot = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    x2plot = np.linspace(min(X[:, 1]), max(X[:, 1]), 100)
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = model.predict(np.column_stack([X1.reshape(-1), X2.reshape(-1)])).reshape(X1.shape)

    # Plot the SVM boundary
    plt.contour(X1, X2, vals, 'b', levels=[0])
    plt.show()


def dataset3Params(X, y, Xval, yval):
    """returns your choice of C and sigma for Part 3 of the exercise
    where you select the optimal (C, sigma) learning parameters to use for SVM
    with RBF kernel

    [C, sigma] = dataset3Params(X, y, Xval, yval) returns your choice of C and
    sigma. You should complete this function to return the optimal C and
    sigma based on a cross-validation set.
    :param X:
    :param y:
    :param Xval:
    :param yval:
    :return:
    """
    values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    max_accu = 0

    from sklearn import svm
    clf = svm.SVC()
    for C in values:
        clf.C = C
        for sigma in values:
            clf.kernel = lambda x1, x2: gaussianKernel(x1, x2, sigma)
            clf.fit(X, y)
            pred = clf.predict(Xval)
            accu = (pred == yval).mean()
            if accu > max_accu:
                max_accu = accu
                best_C = C
                best_sigma = sigma

    return best_C, best_sigma
