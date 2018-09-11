import numpy as np
from matplotlib import pyplot as plt

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


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_):
    """Implements the neural network cost function for a two layer
    neural network which performs classification

    [J grad] = nnCostFunction(nn_params, hidden_layer_size, num_labels, ...
    X, y, lambda) computes the cost and gradient of the neural network.
    :param nn_params: "unrolled" parameters for the neural network,
    need to be converted back into the weight matrices.
    :param input_layer_size:
    :param hidden_layer_size:
    :param num_labels:
    :param X:
    :param y:
    :param lambda_:
    :return: grad should be a "unrolled" vector of the
    partial derivatives of the neural network.
    """
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape((hidden_layer_size, input_layer_size + 1))
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape((num_labels, hidden_layer_size + 1))

    # Setup some useful variables
    m = X.shape[0]

    # Add ones to the X data matrix
    X = np.insert(X, 0, 1, axis=1)

    z2 = np.matmul(X, Theta1.transpose())
    a2 = sigmoid(z2)
    a2 = np.insert(a2, 0, 1, axis=1)
    z3 = np.matmul(a2, Theta2.transpose())
    a3 = sigmoid(z3)

    y_one_hot = np.zeros_like(a3)
    for i in range(m):
        y_one_hot[i, y[i] - 1] = 1

    ones = np.ones_like(a3)
    A = (np.matmul(y_one_hot.transpose(), np.log(a3)) + np.matmul((ones - y_one_hot).transpose(), np.log(ones - a3)))
    J = -1 / m * A.trace()
    J += lambda_ / (2 * m) * (np.sum(Theta1[:, 1:] ** 2) + np.sum(Theta2[:, 1:] ** 2))

    delta3 = a3 - y_one_hot
    delta2 = np.matmul(delta3, Theta2[:, 1:]) * sigmoidGradient(z2)
    Theta2_grad = np.matmul(a2.transpose(), delta3).transpose()
    Theta1_grad = np.matmul(X.transpose(), delta2).transpose()

    Theta1_grad[:, 1:] += lambda_ * Theta1[:, 1:]
    Theta2_grad[:, 1:] += lambda_ * Theta2[:, 1:]
    Theta1_grad /= m
    Theta2_grad /= m
    grad = np.concatenate([Theta1_grad.reshape(-1), Theta2_grad.reshape(-1)])
    return J, grad


def sigmoidGradient(z):
    """returns the gradient of the sigmoid function evaluated at z

    g = sigmoidGradient(z) computes the gradient of the sigmoid function
    evaluated at z. This should work regardless if z is a matrix or a
    vector.
    :param z:
    :return:
    """
    return sigmoid(z) * (1 - sigmoid(z))


def randInitializeWeights(L_in, L_out):
    """Randomly initialize the weights of a layer with L_in
    incoming connections and L_out outgoing connections

    W = randInitializeWeights(L_in, L_out) randomly initializes the weights
    of a layer with L_in incoming connections and L_out outgoing
    connections.
    :param L_in:
    :param L_out:
    :return: W: a matrix of size(L_out, 1 + L_in) as
    the first column of W handles the "bias" terms
    """
    epsilon_init = np.sqrt(6 / (L_in + L_out))
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return W


def checkNNGradients(lambda_ = 0):
    """Creates a small neural network to check the backpropagation gradients

    checkNNGradients(lambda) Creates a small neural network to check the
    backpropagation gradients, it will output the analytical gradients
    produced by your backprop code and the numerical gradients (computed
    using computeNumericalGradient). These two gradient computations should
    result in very similar values.
    :param lambda_:
    :return:
    """
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    # Reusing debugInitializeWeights to generate X
    X = debugInitializeWeights(m, input_layer_size - 1)
    y = 1 + np.arange(1, m + 1) % num_labels

    # Unroll parameters
    nn_params = np.concatenate([Theta1.reshape(-1), Theta2.reshape(-1)])

    # Short hand for cost function
    costFunc = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lambda_)

    cost, grad = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)

    # Visually examine the two gradient computations.The two columns
    # you get should be very similar.
    print(np.column_stack([numgrad, grad]))
    print('The above two columns you get should be very similar.\n'
          '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    # Evaluate the norm of the difference between two solutions.
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001
    # in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = np.std(numgrad - grad) / np.std(numgrad + grad)

    print('If your backpropagation implementation is correct, then \n'
          'the relative difference will be small (less than 1e-9). \n'
          f'\nRelative Difference: {diff:g}')


def debugInitializeWeights(fan_out, fan_in):
    """Initialize the weights of a layer with fan_in
    incoming connections and fan_out outgoing connections using a fixed
    strategy, this will help you later in debugging

    W = debugInitializeWeights(fan_in, fan_out) initializes the weights
    of a layer with fan_in incoming connections and fan_out outgoing
    connections using a fix set of values
    :param fan_out:
    :param fan_in:
    :return: W: a matrix of size(1 + fan_in, fan_out) as
    the first row of W handles the "bias" terms
    """
    # Initialize W using "sin", this ensures that W is always of the same
    # values and will be useful for debugging
    W = np.sin(np.arange(1, fan_out * (1 + fan_in) + 1)).reshape((fan_out, 1 + fan_in)) / 10
    return W


def computeNumericalGradient(J, theta):
    """Computes the gradient using "finite differences"
    and gives us a numerical estimate of the gradient.

    numgrad = computeNumericalGradient(J, theta) computes the numerical
    gradient of the function J around theta. Calling y = J(theta) should
    return the function value at theta.
    :param J:
    :param theta:
    :return: numgrad(i): a numerical approximation of)
    the partial derivative of J with respect to the
    i-th input argument, evaluated at theta.
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
