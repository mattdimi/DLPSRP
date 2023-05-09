import pandas as pd
import datetime as dt
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, Lasso
import seaborn as sns
import numpy as np
import utils
from Model import Model
import matplotlib.pyplot as plt
from tqdm import tqdm
from PrecisionMatrixBuilder import PrecisionMatrixBuilder
from Forecaster import Forecaster
from MyDataLoader import MyDataLoader
from PortfolioConstructor import PortfolioConstructor
from BackTester import BackTester
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer, Baseline
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss, RMSE
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import torch
import pytorch_lightning as pl
from networks import MLP, LSTM, GRU, CNN
from Forecaster import Forecaster
import pickle
import os

def main():

    # Load data
    print("Loading data...")
    dataloader = MyDataLoader()
    full_dataset, stock_names, time_index = dataloader.load_full_dataset_array()
    returns = dataloader.load_returns()
    benchmark_returns = dataloader.load_benchmark(date_index=time_index)
    risk_free = dataloader.load_risk_free_rate(time_index)
    config = utils.load_config()
    alphas = config["alphas"]
    thetas = config["thetas"]
    lambdas = config["lambdas"]
    capital = config["capital"]
    precision_matrix_file_name = config["precision_matrix_file_name"]
    precision_matrix_builder = PrecisionMatrixBuilder(returns = returns, window = 512, alphas = alphas, thetas = thetas)

    print("Building precision matrix...")
    if precision_matrix_file_name in os.listdir("."):
        print("Local version of precision matrix found, loading...")
        with open(precision_matrix_file_name, "rb") as f:
            prec_mat = pickle.load(f)
    else:
        print("Building precision matrix and saving it in local directory...")
        prec_mat = precision_matrix_builder.get_precision_matrix()
        with open(precision_matrix_file_name, "wb") as f:
            pickle.dump(prec_mat, f)

    forecaster = Forecaster(full_dataset, stock_names, time_index, valid_size=66, test_size=66, n_splits=10)
    
    # Define models
    model_regression = Model(name = "MultiLayerPerceptronRegression", type = "regression", model_class= MLP(14, [64], model_type="regression"))
    model_classification = Model(name = "MultiLayerPerceptronClassification", type = "classification", model_class= MLP(14,  [64], model_type="classification"))
    
    models = [model_regression, model_classification]

    # Obtain forecasts and weights
    print("Training and evaluating models...")
    forecasts_val, forecasts_test, weights_valid, weights_test, opt_param_dict, valid_index_full, test_index_full = forecaster.evaluate_test_models(
        models = models,
        prec_mat = prec_mat,
        lambdas = lambdas,
        alphas = alphas,
        thetas = thetas,
        capital = capital,
        stock_returns=returns,
        risk_free=risk_free,
        benchmark_returns=benchmark_returns)
    
    # Backtest
    for model in models:

        print(f"Backtesting {model.name}...")
        backtest = BackTester(weights_test[model.name], returns, capital, risk_free, benchmark=benchmark_returns, name = model.name)
        backtest_statistics = backtest.get_backtest_statistics()
        cum_returns = backtest.get_strategy_cumulative_returns()

        if model.type == "regression":
            cum_returns["MSE"] = (forecasts_test[model.name] - returns.loc[test_index_full, :]).apply(lambda x: x**2).mean(axis = 1)
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            cum_returns.plot(y = [model.name, "CAC 40"], ax = ax1, figsize = (15, 10))
            cum_returns.plot(y = "MSE", ax = ax2, color = "red")
            ax1.set_ylabel("Cumulative returns")
            ax2.set_ylabel("MSE")
        
        elif model.type == "classification":
            cum_returns["Missclassification rate"] = (2*forecasts_test[model.name]-1 != np.sign(returns.loc[test_index_full, :])).abs().mean(axis = 1).cumsum()
            fig, ax1 = plt.subplots()
            ax2 = ax1.twinx()
            cum_returns.plot(y = [model.name, "CAC 40"], ax = ax1, figsize = (15, 10))
            cum_returns.plot(y = "Missclassification rate", ax = ax2, color = "red")
            ax1.set_ylabel("Cumulative returns")
            ax2.set_ylabel("Missclassification rate")
        print(backtest_statistics)

if __name__ == "__main__":
    main()