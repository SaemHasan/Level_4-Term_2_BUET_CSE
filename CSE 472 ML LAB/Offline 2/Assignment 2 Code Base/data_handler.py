import pandas as pd
import numpy as np
import random

def load_dataset():
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    # todo: implement
    X, y = None, None
    df = pd.read_csv('data_banknote_authentication.csv')
    # print(df.head())
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


def shuffle_data(X, y):
    """
    function for shuffling data
    :param X:
    :param y:
    :return:
    """
    # print(X[:10], y[:10])
    full_data = list(zip(X, y))
    random.shuffle(full_data)
    X, y = zip(*full_data)
    # print(X[:10], y[:10])
    X = np.array(X)
    y = np.array(y)
    # print(X[:10], y[:10])
    return X, y


def split_dataset(X, y, test_size, shuffle):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    # todo: implement.
    X_train, y_train, X_test, y_test = None, None, None, None
    if shuffle:
        X, y = shuffle_data(X, y)    
    split_index = int(X.shape[0] * (1 - test_size))
    X_train, y_train = X[:split_index], y[:split_index]
    X_test, y_test = X[split_index:], y[split_index:]
    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # todo: implement
    # https://pynative.com/python-random-sample/
    X_sample, y_sample = None, None
    full_data = list(zip(X, y))
    sample = random.choices(full_data, k=len(full_data))
    X_sample, y_sample = zip(*sample)
    X_sample = np.array(X_sample)
    y_sample = np.array(y_sample)
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample
