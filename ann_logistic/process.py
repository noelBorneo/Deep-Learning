import numpy as np
import pandas as pd

def get_data():

    df = pd.read_csv('ecommerce_data.csv')  # read data from data set
    data = df.as_matrix()

    # split x and y (last column)
    X = data[:, :-1]
    Y = data[:, -1]

    # normalise numeric columns using coefficient of variation
    X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()

