import numpy as np
from process import get_binary_data

# Call function
X, Y = get_binary_data()

# Get dimensionality of the data set
D = X.shape[1]

# Initialize the weights of logistic regression model
W = np.random.randn(D)
b = 0  # bias term

def sigmoid(a):
    """
    Sigmoid activation function
    :param a:
    :return int:
    """
    return 1 / (1 + np.exp(-a))

def forward(X, W, b):
    """
    feed forward function
    :param X: Input
    :param W: Weights
    :param b: Bias term
    :return:
    """
    return sigmoid(X.dot(W) + b)

P_Y_given_X = forward(X, W, b)
predictions = np.round(P_Y_given_X)

def classification_rate(Y, P):
    """
    cross-entropy function
    :param Y: Targets
    :param P: Predictions
    :return: mean-value
    """
    return np.mean(Y == P)

print "Score: ", classification_rate(Y, predictions)
