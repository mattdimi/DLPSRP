import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from scipy.ndimage.interpolation import shift
from BackTester import BackTester

def drawdown(return_series: pd.Series):
    """Takes a time series of asset returns.
       returns its max drawdown
    """
    wealth_index = return_series.cumsum()
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

