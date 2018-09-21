import numpy as np

np.set_printoptions(precision=6)


def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda_):
    """Collaborative filtering cost function

    [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
    num_features, lambda) returns the cost and gradient for the
    collaborative filtering problem.
    :param params:
    :param Y:
    :param R:
    :param num_users:
    :param num_movies:
    :param num_features:
    :param lambda_:
    :return:
    """
    # Unfold the U and W matrices from params
    X = params[:num_movies * num_features].reshape((num_movies, num_features))
    Theta = params[num_movies * num_features:].reshape((num_users, num_features))

    diff = np.matmul(X, Theta.transpose()) - Y
    J = 1 / 2 * (diff ** 2 * R).sum()

    X_grad = np.matmul(diff * R, Theta)
    Theta_grad = np.matmul((diff * R).transpose(), X)

    J += lambda_ / 2 * ((Theta ** 2).sum() + (X ** 2).sum())

    X_grad += lambda_ * X
    Theta_grad += lambda_ * Theta

    grad = np.concatenate([X_grad.reshape(-1), Theta_grad.reshape(-1)])
    return J, grad


def computeNumericalGradient(J, theta):
    """Computes the gradient using "finite differences"
    and gives us a numerical estimate of the gradient.

    numgrad = computeNumericalGradient(J, theta) computes the numerical
    gradient of the function J around theta. Calling y = J(theta) should
    return the function value at theta.
    :param J:
    :param theta:
    :return:
    """
    numgrad = np.zeros_like(theta).reshape(-1)
    perturb = np.zeros_like(theta).reshape(-1)
    e = 1e-4
    for p in range(theta.size):
        # Set perturbation vector
        perturb[p] = e
        loss1, _ = J(theta - perturb.reshape(theta.shape))
        loss2, _ = J(theta + perturb.reshape(theta.shape))
        # Compute Numerical Gradient
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0

    return numgrad.reshape(theta.shape)


def checkCostFunction(lambda_ = 0):
    """Creates a collaborative filering problem
    to check your cost function and gradients

    checkCostFunction(lambda) Creates a collaborative filering problem
    to check your cost function and gradients, it will output the
    analytical gradients produced by your code and the numerical gradients
    (computed using computeNumericalGradient). These two gradient
    computations should result in very similar values.
    :param lambda_:
    :return:
    """
    ## Create small problem
    X_t = np.random.random((4, 3))
    Theta_t = np.random.random((5, 3))

    # Zap out most entries
    Y = np.matmul(X_t, Theta_t.transpose())
    Y[np.random.random(Y.shape) > 0.5] = 0
    R = np.zeros_like(Y)
    R[Y != 0] = 1

    ## Run Gradient Checking
    X = np.random.randn(*X_t.shape)
    Theta = np.random.randn(*Theta_t.shape)
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]

    numgrad = computeNumericalGradient(
        lambda t: cofiCostFunc(t, Y, R, num_users, num_movies, num_features, lambda_),
        np.concatenate([X.reshape(-1), Theta.reshape(-1)]))

    cost, grad = cofiCostFunc(np.concatenate([X.reshape(-1), Theta.reshape(-1)]),
        Y, R, num_users, num_movies, num_features, lambda_)

    print(np.column_stack([numgrad, grad]))
    print(['The above two columns you get should be very similar.\n'
           '(Left-Your Numerical Gradient, Right-Analytical Gradient)'])

    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print('If your cost function implementation is correct, then \n'
          'the relative difference will be small (less than 1e-9). \n'
          f'\nRelative Difference: {diff:g}')


def loadMovieList():
    """reads the fixed movie list in movie.txt and returns a
    cell array of the words

    movieList = loadMovieList() reads the fixed movie list in movie.txt
    and returns a cell array of the words in movieList.
    :return:
    """
    ## Read the fixed movieulary list
    with open('movie_ids.txt', encoding='ISO-8859-1') as fid:
        # Store all movies in cell array movie{}
        n = 1682  # Total number of movies

        movieList = []
        for i in range(n):
            # Read line
            line = fid.readline()
            # Word Index (can ignore since it will be = i)
            idx, movieName = line.split(' ', maxsplit=1)
            # Actual Word
            movieList.append(movieName.strip())

    return movieList


def normalizeRatings(Y, R):
    """Preprocess data by subtracting mean rating for every movie (every row)

    [Ynorm, Ymean] = normalizeRatings(Y, R) normalized Y so that each movie
    has a rating of 0 on average, and returns the mean rating in Ymean.
    :param Y:
    :param R:
    :return:
    """
    m, n = Y.shape
    Ymean = np.zeros(m)
    Ynorm = np.zeros_like(Y)
    for i in range(m):
        idx = R[i] == 1
        Ymean[i] = Y[i, idx].mean()
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]
    return Ynorm, Ymean
