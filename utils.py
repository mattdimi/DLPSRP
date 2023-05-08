import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from scipy.ndimage.interpolation import shift
import json

def drawdown(return_series: pd.Series):
    """Returns the drawdown of a return series

    Args:
        return_series (pd.Series): input return series

    Returns:
        float: drawdown of the return series
    """
    wealth_index = return_series.cumsum()
    previous_peaks = wealth_index.cummax()
    drawdowns = (previous_peaks - wealth_index)/previous_peaks
    return drawdowns.abs().max()

def time_series_cross_validation(dataset, time_index, valid_size = 100, test_size = 30, n_splits = 50):
    """Yields an iterator for the time series validation

    Args:
        dataset (pd.DataFrame | np.array): dataset used
        time_index (list): time index of the dataset
        valid_size (int, optional): fixed validation size. Defaults to 100.
        test_size (int, optional): fixed test size. Defaults to 30.
        n_splits (int, optional): number of splits in the timeseries validation. Defaults to 50.

    Yields:
        tuple: (train data, train time index, validation data, validation time index, test data, test time index)
    """
    tss = TimeSeriesSplit(n_splits = n_splits, test_size = test_size)
    for train_index, test_index in tss.split(dataset):
        train, test = dataset[train_index, :, :], dataset[test_index, :, :]
        train, valid = train[:-valid_size, :, :], train[-valid_size:, :, :]

        time_index_train = time_index[train_index[:-valid_size]]
        time_index_valid = time_index[train_index[-valid_size:]]
        time_index_test = time_index[test_index]
        
        yield train, time_index_train, valid, time_index_valid, test, time_index_test

def load_config():
    """Load configuration file

    Returns:
        dict: configuration file in a dictionary
    """
    with open("config.json") as f:
        config = json.load(f)
    return config

def load_opt_params_tft():
    """Load optimal parameters for each stock for TFT model

    Returns:
        dict: optimal parameters for each stock for TFT model in a dictionary
    """
    with open("opt_params_tft.json") as f:
        opt_params_tft = json.load(f)
    return opt_params_tft

def load_data_for_rnn(X, y, len_train, len_valid, len_test, lookback_window):
    """Method to load data into a np.array which is suitable for recurrent neural networks

    Args:
        X (np.array): features of shape (number of samples, number of features)
        y (np.array): target of shape (number of samples, )
        lookback_window (int): number of previous days to look back
    """
    X_temp = []
    y_temp = []
    for i in range(lookback_window-1, X.shape[0]):
        X_temp.append(X[i-lookback_window+1:i+1,:]) # features already shifted by 1
        y_temp.append(y[i])
    X_train = np.array(X_temp[:len_train-lookback_window+1])
    y_train = np.array(y_temp[:len_train-lookback_window+1])
    X_valid = np.array(X_temp[len_train-lookback_window+1:len_train+len_valid-lookback_window+1])
    y_valid = np.array(y_temp[len_train-lookback_window+1:len_train+len_valid-lookback_window+1])
    X_test = np.array(X_temp[len_train+len_valid-lookback_window+1:])
    y_test = np.array(y_temp[len_train+len_valid-lookback_window+1:])
    return X_train, y_train, X_valid, y_valid, X_test, y_test


