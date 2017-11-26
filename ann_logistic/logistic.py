import numpy as np

# Some data to do logistic regression on
# This is the parameters to set a matrix
N = 100
D = 2

# Creating a normally distributed matrix
X = np.random.rand(N, D)

'''
 Add a bias term or w0
 Could create a 2D weight vector
 through dot product(sum of the products)
 of (X, weights) then add bias
 Probably, add a column of ones to the OG data
 include bias term in weights W
'''

ones = np.array([[1]*N]).T # need 2D but array in numpy is 1D

# Concatenate vector of ones to OG data set
Xb = np.concatenate((ones, X), axis=1)

# Randomly initialise the weight vector
w = np.random.randn(D + 1)

# Perform dot product on vectors Xb and weight
z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

print sigmoid(z)

