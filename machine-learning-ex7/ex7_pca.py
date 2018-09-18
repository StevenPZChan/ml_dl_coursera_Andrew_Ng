import numpy as np
from matplotlib import pyplot as plt

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
    mu = X.mean(axis=0)
    X_norm = X - mu

    sigma = X_norm.std(axis=0)
    X_norm = X_norm / sigma
    return X_norm, mu, sigma


def pca(X):
    """Run principal component analysis on the dataset X

    [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
    :param X:
    :return: the eigenvectors U, the eigenvalues (on diagonal) in S
    """
    # Useful values
    m, n = X.shape

    sigma = 1 / m * np.matmul(X.transpose(), X)
    u, s, _ = np.linalg.svd(sigma)
    U = u
    S = np.diag(s)
    return U, S


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


def projectData(X, U, K):
    """Computes the reduced data representation when projecting only
    on to the top k eigenvectors

    Z = projectData(X, U, K) computes the projection of
    the normalized inputs X into the reduced dimensional space spanned by
    the first K columns of U. It returns the projected examples in Z.
    :param X:
    :param U:
    :param K:
    :return:
    """
    Z = np.matmul(X, U[:, :K])
    return Z


def recoverData(Z, U, K):
    """Recovers an approximation of the original data when using the
    projected data

    X_rec = recoverData(Z, U, K) recovers an approximation the
    original data that has been reduced to K dimensions. It returns the
    approximate reconstruction in X_rec.
    :param Z:
    :param U:
    :param K:
    :return:
    """
    X_rec = np.matmul(Z, U.transpose()[:K])
    return X_rec


def displayData(X, example_width = None):
    """Display 2D data in a nice grid

    [h, display_array] = displayData(X, example_width) displays 2D data
    stored in X in a nice grid.
    :param X:
    :param example_width:
    :return: the figure handle h and the displayed array if requested.
    """
    # Set example_width automatically if not passed in
    if example_width is None:
        example_width = int(round(np.sqrt(X.shape[1])))

    # Gray Image
    colormap = 'gray'

    # Compute rows, cols
    m, n = X.shape
    example_height = int(n / example_width)

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    # Between images padding
    pad = 1

    # Setup blank display
    display_array = -np.ones((pad + display_rows * (example_height + pad), pad + display_cols * (example_width + pad)))

    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m:
                break
            # Copy the patch

            # Get the max value of the patch
            max_val = max(abs(X[curr_ex, :]))
            sample_y = pad + j * (example_height + pad)
            sample_x = pad + i * (example_width + pad)
            sample_pixel = X[curr_ex, :].reshape((example_height, example_width), order='F') / max_val
            display_array[sample_y:sample_y + example_height, sample_x:sample_x + example_width] = sample_pixel
            curr_ex += 1
        if curr_ex > m:
            break

    # Display Image
    h = plt.imshow(display_array, cmap=colormap)
    # Do not show axis
    plt.axis('off')

    # plt.show()
    return h, display_array
