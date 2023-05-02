import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from scipy.ndimage.interpolation import shift
import json
from BackTester import BackTester

def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns.
       returns its max drawdown
    """
    wealth_index = return_series.cumsum()
    previous_peaks = wealth_index.cummax()
    drawdowns = (previous_peaks - wealth_index)/previous_peaks
    return drawdowns.abs().max()

def time_series_cross_validation(dataset, time_index, valid_size = 100, test_size = 30, n_splits = 50):
    tss = TimeSeriesSplit(n_splits = n_splits, test_size = test_size)
    for train_index, test_index in tss.split(dataset):
        train, test = dataset[train_index, :, :], dataset[test_index, :, :]
        train, valid = train[:-valid_size, :, :], train[-valid_size:, :, :]

        time_index_train = time_index[train_index[:-valid_size]]
        time_index_valid = time_index[train_index[-valid_size:]]
        time_index_test = time_index[test_index]
        
        yield train, time_index_train, valid, time_index_valid, test, time_index_test

def run_backtests(constructed_portfolios, dataloader, time_index, models, lambdas, alphas, thetas):
    returns = dataloader.load_returns()
    risk_free = dataloader.load_risk_free_rate(time_index)
    capital = 1
    backtest_results = {}
    backtest_performances = {}
    for model in models:
        model_name = model.name
        weights, time_index, stocks = constructed_portfolios[model_name]
        if model.type == "classification":
            weights_df = pd.DataFrame(weights, index = time_index, columns = stocks)
            backtester = BackTester(weights_df, returns, capital, risk_free)
            backtest_stats = backtester.get_backtest_statistics()
            _, backtest_cumulative_performance = backtester.run_backtest()
            backtest_results[(model_name, "", "", "")] = backtest_stats
            backtest_performances[(model_name, "", "", "")] = backtest_cumulative_performance

        elif model.type == "regression":
            for i, lam in enumerate(lambdas):
                for j, alpha in enumerate(alphas):
                    for k, theta in enumerate(thetas):
                        weights_ijk= weights[i][j][k]
                        weights_df = pd.DataFrame(weights_ijk, index = time_index, columns = stocks)
                        backtester = BackTester(weights_df, returns, capital, risk_free)
                        _, backtest_cumulative_performance = backtester.run_backtest()
                        backtest_stats = backtester.get_backtest_statistics()
                        backtest_results[(model_name, lam, alpha, theta)] = backtest_stats
                        backtest_performances[(model_name, lam, alpha, theta)] = backtest_cumulative_performance

    backtest_results = pd.concat(backtest_results, axis = 1)
    backtest_results.columns.names = ["Model", "Lambda", "Alpha", "Theta"]

    backtest_performances = pd.concat(backtest_performances, axis = 1)
    backtest_performances.columns.names = ["Model", "Lambda", "Alpha", "Theta"]
    
    return backtest_results, backtest_performances

def load_config():
    with open("config.json") as f:
        config = json.load(f)
    return config

def load_opt_params_tft():
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


