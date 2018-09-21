import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(precision=6)


def estimateGaussian(X):
    """This function estimates the parameters of a
    Gaussian distribution using the data in X

    [mu sigma2] = estimateGaussian(X),
    The input X is the dataset with each n-dimensional data point in one row
    The output is an n-dimensional vector mu, the mean of the data set
    and the variances sigma^2, an n x 1 vector
    :param X:
    :return:
    """
    # Useful variables
    m, n = X.shape

    mu = X.mean(axis=0)
    sigma2 = ((X - mu) ** 2).mean(axis=0)
    return mu, sigma2


def multivariateGaussian(X, mu, Sigma2):
    """Computes the probability density function of the
    multivariate gaussian distribution.

    p = multivariateGaussian(X, mu, Sigma2) Computes the probability
    density function of the examples X under the multivariate gaussian
    distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
    treated as the covariance matrix. If Sigma2 is a vector, it is treated
    as the \sigma^2 values of the variances in each dimension (a diagonal
    covariance matrix)
    :param X:
    :param mu:
    :param Sigma2:
    :return:
    """
    k = len(mu)

    if len(Sigma2.shape) == 1:
        Sigma2 = np.diag(Sigma2)

    X = X - mu.reshape(1, -1)
    p = (2 * np.pi) ** (- k / 2) * np.linalg.det(Sigma2) ** (-0.5) * np.exp(
        -0.5 * np.diag(np.matmul(np.matmul(X, np.linalg.pinv(Sigma2)), X.transpose())))
    return p


def visualizeFit(X, mu, sigma2):
    """Visualize the dataset and its estimated distribution.

    visualizeFit(X, p, mu, sigma2) This visualization shows you the
    probability density function of the Gaussian distribution. Each example
    has a location (x1, x2) that depends on its feature values.
    :param X:
    :param mu:
    :param sigma2:
    :return:
    """
    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    Z = multivariateGaussian(np.column_stack([X1.reshape(-1), X2.reshape(-1)]), mu, sigma2)
    Z = Z.reshape(X1.shape)

    plt.plot(X[:, 0], X[:, 1], 'bx')
    # Do not plot if there are infinities
    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X1, X2, Z, 10.0 ** np.arange(-20, 1, 3))


def selectThreshold(yval, pval):
    """Find the best threshold (epsilon) to use for selecting outliers

    [bestEpsilon bestF1] = selectThreshold(yval, pval) finds the best
    threshold to use for selecting outliers based on the results from a
    validation set (pval) and the ground truth (yval).
    :param yval:
    :param pval:
    :return:
    """
    bestEpsilon = np.nan
    bestF1 = 0

    stepsize = (max(pval) - min(pval)) / 1000
    for epsilon in np.arange(min(pval) + stepsize, max(pval) + stepsize, stepsize):
        predictions = (pval < epsilon).astype(int)
        precision = ((predictions == 1) & (yval == 1)).sum() / (predictions == 1).sum()
        recall = ((predictions == 1) & (yval == 1)).sum() / (yval == 1).sum()
        F1 = 2 * precision * recall / (precision + recall)

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1
