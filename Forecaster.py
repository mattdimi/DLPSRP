import utils
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from BackTester import BackTester
from PortfolioConstructor import PortfolioConstructor
from torch.utils.data import DataLoader
from MyDataset import MyDataset
import torch

class Forecaster:

    def __init__(self, dataset, stock_names, time_index, valid_size, test_size, n_splits = 10):
        self.dataset = dataset
        self.stock_names = stock_names
        self.time_index = time_index
        self.valid_size = valid_size
        self.test_size = test_size
        self.n_splits = n_splits

    def get_tscv_splits(self):
        return utils.time_series_cross_validation(self.dataset, self.time_index, self.valid_size, self.test_size, self.n_splits)

    def evaluate_test_models(self, models, prec_mat, alphas, thetas, lambdas, capital, stock_returns, risk_free, benchmark_returns = None):
        """Method to fine-tune the models of the validation set and then test them on the test set.

        Args:
            models (list[Model]): _description_
            prec_mat (dict): (key = dates, value = len(alphas) x len(thetas) x returns.shape[1] x returns.shape[1] arrays) 
            alphas (list): alphas used in GLASSO
            thetas (list): thetas used in EPO
            lambdas (list): lambdas used in the portfolio construction
            capital (float): capital used in the backtest
            stock_returns (pd.DataFrame): stock_returns used in the backtest
            risk_free (pd.Series): risk_free asset returns
            benchmark_returns (pd.Series): benchmark returns

        Returns:
            tuple: validation forecasts, test forecasts, validation weights, test weights, optimal parameters, validation time index, test time index
        """
        forecasts_val = defaultdict(list)
        forecasts_test = defaultdict(list)

        valid_index_full = []
        test_index_full = []

        opt_param_dict = defaultdict(list)

        weights_valid = defaultdict(list)
        weights_test = defaultdict(list)

        stock_names = self.stock_names
        time_series_split = self.get_tscv_splits()

        for batch in tqdm(time_series_split):

            train, train_index, valid, valid_index, test, test_index = batch

            valid_index_full.append(valid_index)
            test_index_full.append(test_index)

            n_stocks = train.shape[1]

            for model in models:
                
                model_pred_valid = dict()
                model_pred_test = dict()

                cls = model.cls

                for stock in range(n_stocks):

                    stock_name = stock_names[stock]

                    stock_train = train[:, stock, :]
                    stock_valid = valid[:, stock, :]
                    stock_test = test[:, stock, :]

                    y_reg_train = stock_train[:, 0]
                    y_classification_train = stock_train[:, 1]
                    X_train = stock_train[:, 2:]
                    
                    y_reg_valid = stock_valid[:, 0]
                    y_classification_valid = stock_valid[:, 1]
                    X_valid = stock_valid[:, 2:]

                    y_reg_test = stock_test[:, 0]
                    y_classification_test = stock_test[:, 1]
                    X_test = stock_test[:, 2:]

                    scaler_X = StandardScaler()
                    scaler_X.fit(X_train)
                    X_train = scaler_X.transform(X_train)
                    X_valid = scaler_X.transform(X_valid)
                    X_test = scaler_X.transform(X_test)
                    
                    scaler_y = StandardScaler()
                    scaler_y.fit(y_reg_train.reshape(-1, 1))
                    y_reg_train = scaler_y.transform(y_reg_train.reshape(-1, 1)).reshape(-1)
                    y_reg_valid = scaler_y.transform(y_reg_valid.reshape(-1, 1)).reshape(-1)
                    y_reg_test = scaler_y.transform(y_reg_test.reshape(-1, 1)).reshape(-1)

                    y_train, y_valid, y_test = y_reg_train, y_reg_valid, y_reg_test if model.type == "regression" else y_classification_train, y_classification_valid, y_classification_test

                    X_full = np.vstack([X_train, X_valid, X_test])
                    y_full = np.vstack([y_train, y_valid, y_test])
                    
                    if model.is_rnn:
                        X_train, y_train, X_valid, y_valid, X_test, y_test = utils.load_data_for_rnn(X_full, y_full, len(train_index), len(valid_index), len(test_index), lookback_window=60)
                    
                    dataset_train = MyDataset(X_train, y_train)
                    dataset_valid = MyDataset(X_valid, y_valid)
                    dataset_test = MyDataset(X_test, y_test)

                    train_dataloader = DataLoader(dataset=dataset_train, batch_size=64, shuffle=False)
                    valid_dataloader = DataLoader(dataset=dataset_valid, batch_size=64, shuffle=False)
                    test_dataloader = DataLoader(dataset=dataset_test, batch_size=64, shuffle=False)
                    
                    fitted_model = cls.fit(train_dataloader, valid_dataloader) if isinstance(cls, torch.nn.Module) else cls.fit(X_train, y_reg_train if model.type == "regression" else y_classification_train)

                    y_valid_pred = fitted_model.predict(valid_dataloader) if isinstance(cls, torch.nn.Module) else fitted_model.predict(X_valid)
                    y_test_pred = fitted_model.predict(test_dataloader) if isinstance(cls, torch.nn.Module) else fitted_model.predict(X_test)
                    
                    model_pred_valid[stock_name] = pd.Series(y_valid_pred, index = valid_index)
                    model_pred_test[stock_name] = pd.Series(y_test_pred, index = test_index)

                model_pred_valid = pd.concat(model_pred_valid, axis = 1)

                model_pred_test = pd.concat(model_pred_test, axis = 1)

                forecasts_val[model.name] = forecasts_val[model.name] + [model_pred_valid]
                forecasts_test[model.name] = forecasts_test[model.name] + [model_pred_test]

                if model.type == "regression":
                    # validate portfolio optimization parameters
                    pc_valid = PortfolioConstructor(
                        model = model,
                        stock_names = stock_names,
                        model_forecasts = model_pred_valid,
                        precision_matrices = prec_mat,
                        alphas = alphas,
                        thetas = thetas,
                        lambdas = lambdas
                        )
                    portfolios_valid, date_index_valid, stocks = pc_valid.get_portfolios()
                    backtest_results_model = {}
                    for i, lam in enumerate(lambdas):
                        for j, alpha in enumerate(alphas):
                            for k, theta in enumerate(thetas):
                                weights_valid_model = portfolios_valid[i, j, k, :, :]
                                weights_valid_model = pd.DataFrame(weights_valid_model, index = date_index_valid, columns = stocks)
                                backtester_valid = BackTester(weights_valid_model, stock_returns, capital, risk_free, benchmark_returns)
                                backtest_results = backtester_valid.get_backtest_statistics()
                                backtest_results_model[(i, j, k)] = backtest_results
                    backtest_results_model = pd.concat(backtest_results_model, axis = 1).T[["Yearly Sharpe", "Yearly Calmar", "Yearly Sortino"]].mean(axis = 1)
                    best_params_idx = backtest_results_model.idxmax(axis = 0)
                    best_lambda_idx, best_alpha_idx, best_theta_idx = best_params_idx
                    best_lambda = lambdas[best_lambda_idx]
                    best_alpha = alphas[best_alpha_idx]
                    best_theta = thetas[best_theta_idx]
                    best_params_df = pd.DataFrame(np.nan, index = date_index_valid, columns = ["lambda", "alpha", "theta"])
                    best_params_df.loc[:, "lambda"] = best_lambda
                    best_params_df.loc[:, "alpha"] = best_alpha
                    best_params_df.loc[:, "theta"] = best_theta
                    opt_param_dict[model.name] = opt_param_dict[model.name] + [best_params_df]

                    portfolio_valid = pd.DataFrame(portfolios_valid[best_lambda_idx, best_alpha_idx, best_theta_idx, :, :], index = date_index_valid, columns = stocks)
                    weights_valid[model.name] = weights_valid[model.name] + [portfolio_valid]

                    pc_test = PortfolioConstructor(
                        model = model,
                        stock_names = stock_names,
                        model_forecasts = model_pred_test, 
                        precision_matrices = prec_mat, 
                        alphas = [best_alpha],
                        thetas = [best_theta],
                        lambdas = [best_lambda]
                        )
                    portfolio_test, date_index_test, stocks = pc_test.get_portfolios()
                    portfolio_test = portfolio_test[0][0][0]
                    portfolio_test = pd.DataFrame(portfolio_test, index = date_index_test, columns = stocks)
                    weights_test[model.name] = weights_test[model.name] + [portfolio_test]
                
                elif model.type == "classification":
                    
                    pc_valid = PortfolioConstructor(
                        model = model,
                        stock_names = stock_names,
                        model_forecasts = model_pred_valid,
                        precision_matrices = None,
                        alphas = None,
                        thetas = None,
                        lambdas = None
                        )
                    portfolios_valid, date_index_valid, stocks_valid = pc_valid.get_portfolios()
                    portfolio_valid = pd.DataFrame(portfolios_valid, index = date_index_valid, columns = stocks_valid)
                    weights_valid[model.name] = weights_valid[model.name] + [portfolio_valid]

                    pc_test = PortfolioConstructor(
                        model = model,
                        stock_names = stock_names,
                        model_forecasts = model_pred_test,
                        precision_matrices = None,
                        alphas = None,
                        thetas = None,
                        lambdas = None
                        )
                    portfolio_test, date_index_test, stocks_test = pc_test.get_portfolios()
                    portfolio_test = pd.DataFrame(portfolio_test, index = date_index_test, columns = stocks_test)
                    weights_test[model.name] = weights_test[model.name] + [portfolio_test]
        
        for model in models:
            forecasts_val[model.name] = pd.concat(forecasts_val[model.name], axis = 0)
            forecasts_test[model.name] = pd.concat(forecasts_test[model.name], axis = 0)
            weights_valid[model.name] = pd.concat(weights_valid[model.name], axis = 0)
            weights_test[model.name] = pd.concat(weights_test[model.name], axis = 0)
        
        for model in models:
            if model.type == "regression":
                opt_param_dict[model.name] = pd.concat(opt_param_dict[model.name], axis = 0)
        
        valid_index_full = np.concatenate(valid_index_full, axis = 0)
        test_index_full = np.concatenate(test_index_full, axis = 0)

        return forecasts_val, forecasts_test, weights_valid, weights_test, opt_param_dict, valid_index_full, test_index_full


        
        
