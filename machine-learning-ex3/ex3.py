import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize as opt

np.set_printoptions(precision=6)


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

    plt.show()
    return h, display_array


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
    m = len(y)  # number of training examples

    h_theta = sigmoid(np.matmul(X, theta))
    ones = np.ones(m)
    J = -1 / m * (np.matmul(y, np.log(h_theta)) + np.matmul(ones - y, np.log(ones - h_theta)))
    grad = 1 / m * np.matmul(X.transpose(), h_theta - y)
    return J, grad


def lrCostFunction(theta, X, y, lambda_):
    """Compute cost and gradient for logistic regression with regularization

    J = lrCostFunction(theta, X, y, lambda) computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    :param theta:
    :param X:
    :param y:
    :param lambda_:
    :return:
    """
    # Initialize some useful values
    m = len(y)  # number of training examples
    J, grad = costFunction(theta, X, y)
    J += lambda_ / (2 * m) * np.matmul(theta[1:], theta[1:])
    grad[1:] += lambda_ / m * theta[1:]
    return J, grad


def oneVsAll(X, y, num_labels, lambda_):
    """Trains multiple logistic regression classifiers and returns all
    the classifiers in a matrix all_theta, where the i-th row of all_theta
    corresponds to the classifier for label i

    [all_theta] = oneVsAll(X, y, num_labels, lambda) trains num_labels
    logistic regression classifiers
    :param X:
    :param y:
    :param num_labels:
    :param lambda_:
    :return: each of these classifiers in a matrix all_theta,
    where the i-th row of all_theta corresponds to the classifier for label i
    """
    # Some useful variables
    m, n = X.shape

    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data matrix
    X = np.insert(X, 0, 1, axis=1)

    # Set Initial theta
    initial_theta = np.zeros(n + 1)

    # Set options for fminunc
    fun = lambda theta, X, y, lambda_: lrCostFunction(theta, X, y, lambda_)[0]
    jac = lambda theta, X, y, lambda_: lrCostFunction(theta, X, y, lambda_)[1]
    options = {'disp': True, 'maxiter': 400}

    for c in range(num_labels):
        # Run fmincg to obtain the optimal theta
        args = (X, (y == c + 1).astype(np.int), lambda_)
        res = opt.minimize(fun, initial_theta, args=args, method='CG', jac=jac, options=options)
        all_theta[c, :] = res.x

    return all_theta


def predictOneVsAll(all_theta, X):
    """Predict the label for a trained one-vs-all classifier. The labels
    are in the range 1..K, where K = size(all_theta, 1).

    p = predictOneVsAll(all_theta, X) will return a vector of predictions
    for each example in the matrix X.
    :param all_theta: a matrix where the i-th row is a trained logistic
    regression theta vector for the i-th class.
    :param X: contains the examples in rows.
    :return: a vector
    of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
    for 4 examples)
    """
    m = X.shape[0]
    num_labels = all_theta.shape[0]

    # Add ones to the X data matrix
    X = np.insert(X, 0, 1, axis=1)

    prob = sigmoid(np.matmul(X, all_theta.transpose()))
    p = prob.argmax(axis=1) + 1
    return p


def predict(Theta1, Theta2, X):
    """Predict the label of an input given a trained neural network

    p = predict(Theta1, Theta2, X) outputs the predicted label of X given the
    trained weights of a neural network (Theta1, Theta2)
    :param Theta1:
    :param Theta2:
    :param X:
    :return:
    """
    # Useful values
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    # Add ones to the X data matrix
    X = np.insert(X, 0, 1, axis=1)

    a2 = sigmoid(np.matmul(X, Theta1.transpose()))
    a2 = np.insert(a2, 0, 1, axis=1)
    a3 = sigmoid(np.matmul(a2, Theta2.transpose()))
    p = a3.argmax(axis=1) + 1
    return p
