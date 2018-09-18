import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(precision=6)


def findClosestCentroids(X, centroids):
    """computes the centroid memberships for every example

    idx = findClosestCentroids (X, centroids) returns the closest centroids
    in idx for a dataset X where each row is a single example. idx = m x 1
    vector of centroid assignments (i.e. each entry in range [1..K])
    :param X:
    :param centroids:
    :return:
    """
    # Set K
    K = centroids.shape[0]
    m = X.shape[0]

    distance = np.zeros((m, K))
    for i in range(K):
        distance[:, i] = np.diag(np.matmul(X - centroids[i], (X - centroids[i]).transpose()))

    idx = distance.argmin(axis=1)
    return idx


def computeCentroids(X, idx, K):
    """returns the new centroids by computing the means of the
    data points assigned to each centroid.

    centroids = computeCentroids(X, idx, K) returns the new centroids by
    computing the means of the data points assigned to each centroid. It is
    given a dataset X where each row is a single data point, a vector
    idx of centroid assignments (i.e. each entry in range [1..K]) for each
    example, and K, the number of centroids. You should return a matrix
    centroids, where each row of centroids is the mean of the data points
    assigned to it.
    :param X:
    :param idx:
    :param K:
    :return:
    """
    # Useful variables
    m, n = X.shape

    centroids = np.zeros((K, n))
    for i in range(K):
        centroids[i] = X[idx == i].mean(axis=0)
    return centroids


def plotDataPoints(X, idx, K):
    """plots data points in X, coloring them so that those with the same
    index assignments in idx have the same color

    plotDataPoints(X, idx, K) plots data points in X, coloring them so that those
    with the same index assignments in idx have the same color
    :param X:
    :param idx:
    :param K:
    :return:
    """
    # Create palette
    palette = plt.cm.get_cmap('rainbow', K)
    # colors = palette[idx]

    # Plot the data
    plt.scatter(X[:, 0], X[:, 1], 15, idx, cmap=palette)


def drawLine(p1, p2, *args, **kwargs):
    """Draws a line from point p1 to point p2

    drawLine(p1, p2) Draws a line from point p1 to point p2 and holds the
    current figure
    :param p1:
    :param p2:
    :param args:
    :return:
    """
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], *args, **kwargs)


def plotProgresskMeans(X, centroids, previous, idx, K, i):
    """a helper function that displays the progress of
    k-Means as it is running. It is intended for use only with 2D data.

    plotProgresskMeans(X, centroids, previous, idx, K, i) plots the data
    points with colors assigned to each centroid. With the previous
    centroids, it also plots a line between the previous locations and
    current locations of the centroids.
    :param X:
    :param centroids:
    :param previous:
    :param idx:
    :param K:
    :param i:
    :return:
    """
    # Plot the examples
    plotDataPoints(X, idx, K)

    # Plot the centroids as black x's
    plt.plot(centroids[:, 0], centroids[:, 1], 'x', markeredgecolor='k', markersize=10, markeredgewidth=3)

    # Plot the history of the centroids with lines
    for j in range(centroids.shape[0]):
        drawLine(centroids[j], previous[j])

    # Title
    plt.title(f'Iteration number {i + 1}')


def runkMeans(X, initial_centroids, max_iters, plot_progress = False):
    """runs the K-Means algorithm on data matrix X, where each row of X
    is a single example

    [centroids, idx] = runkMeans(X, initial_centroids, max_iters, ...
    plot_progress) runs the K-Means algorithm on data matrix X, where each
    row of X is a single example.
    :param X:
    :param initial_centroids: the initial centroids
    :param max_iters: the total number of interactions of K-Means to execute
    :param plot_progress: a true/false flag that
    indicates if the function should also plot its progress as the
    learning happens. This is set to false by default.
    :return: centroids: a Kxn matrix of the computed centroids and idx, a m x 1
    vector of centroid assignments (i.e. each entry in range [1..K])
    """
    # Plot the data if we are plotting progress
    if plot_progress:
        plt.figure()

    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids[:]
    previous_centroids = centroids[:]
    idx = np.zeros(m)

    # Run K-Means
    for i in range(max_iters):

        # Output progress
        print(f'K-Means iteration {i + 1}/{max_iters}...')

        # For each example in X, assign it to the closest centroid
        idx = findClosestCentroids(X, centroids)

        # Optionally, plot progress here
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids[:]
            input('Press enter to continue.')

        # Given the memberships, compute new centroids
        centroids = computeCentroids(X, idx, K)

    # Hold off if we are plotting progress
    if plot_progress:
        plt.show()

    return centroids, idx


def kMeansInitCentroids(X, K):
    """This function initializes K centroids that are to be
    used in K-Means on the dataset X

    centroids = kMeansInitCentroids(X, K) returns K initial centroids to be
    used with the K-Means on the dataset X
    :param X:
    :param K:
    :return:
    """
    centroids = X[np.random.permutation(X.shape[0])[:K]]
    return centroids
