import numpy as np
import pandas as pd

def get_data():

    df = pd.read_csv('ecommerce_data.csv')  # read data from data set
    data = df.as_matrix()

    # split x and y (last column)
    X = data[:, :-1]  # everything except the last column
    Y = data[:, -1]  # everything in the last column
    # print X[:, 1] # debugging

    # normalise numeric columns using coefficient of variation
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()  # n_products_viewed
    X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()  # visit_duration

    # normalise categorical column (time_of_day)
    N, D = X.shape  # returns parameters of X
    print 'N:', N
    print 'D:', D

    # X2 has 8 columns. first 5 from X and 3 additional for categorical values
    X2 = np.zeros((N, D+3))  # D+3 for having 4 categorical values (0 / 1 / 2 / 3)
    print X2
    X2[:, 0:(D-1)] = X[:, 0:(D-1)]  # copies X values to X2 except for time of day


    ''' 
    one-hot encoding for 4 columns (X)
    * * * * * * * * * *
     | 0 | 1 | 2 | 3 |
     -----------------
     | 1 | 0 | 0 | 0 |
     | 0 | 1 | 0 | 0 |
     | 0 | 0 | 1 | 0 |
     | 0 | 0 | 0 | 1 |
    * * * * * * * * * * 
    '''
    for n in xrange(N):
        t = int(X[n, D-1])
        X2[n, t+D-1] = 1
    print X2
    # Another method
    # Z = np.zeros(N, 4)
    # Z[np.arrange(N), X[:, D-1].astype(np.int32)] = 1
    # X2[:, -4] = Z
    # assert (np.abs(X2[:, -4:] - Z).sum() < 10e-10)

    return X2, Y

def get_binary_data():
    X, Y = get_data()
    X2 = X[Y <= 1]
    Y2 = Y[Y <= 1]

    return X2, Y2

get_data()
