import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit

def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns.
       returns its max drawdown
    """
    wealth_index = (1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return drawdowns.abs().max()

def time_series_cross_validation(dataset, time_index, valid_size = 100, test_size = 30, n_splits = 50):
    tss = TimeSeriesSplit(n_splits = n_splits, test_size = test_size)
    for train_index, test_index in tss.split(dataset):
        train, test = dataset[train_index, :, :], dataset[test_index, :, :]
        train, valid = train[:-valid_size, :, :], train[-valid_size:, :, :]

        time_index_train = time_index[train_index]
        time_index_valid = time_index[train_index[-valid_size:]]
        time_index_test = time_index[test_index]
        
        yield train, time_index_train, valid, time_index_valid, test, time_index_test