import numpy as np
import pandas as pd
from tqdm import tqdm

class PortfolioConstructor:

    def __init__(self, model, stock_names, model_forecasts, precision_matrices, alphas, thetas, lambdas):
        """Portfolio construction class. Support regression and classification models.

        Args:
            model (Model): Model object.
            stock_names (list[str]): Universe of stocks considered in the forecasts.
            model_forecasts (pd.DataFrame): Forecasts of the model for each stock.
            precision_matrices (dict): Precision matrices of the model. Only required for regression models.
            alphas (list[float]): Set of alphas to be considered in the portfolio construction. Only required for regression models.
            thetas (list[float]): Set of thetas to be considered in the portfolio construction. Only required for regression models.
            lambdas (list[float]): Set of lambdas to be considered in the portfolio construction. Only required for regression models.
        """
        self.model = model
        self.stock_names = stock_names
        self.model_forecasts = model_forecasts
        self.precision_matrices = precision_matrices
        self.alphas = alphas
        self.thetas = thetas
        self.lambdas = lambdas
    
    def get_portfolios(self):
        
        model = self.model

        model_weights = {}

        columns = self.stock_names.to_list() + ["IRX"]
        
        n_dates = len(self.model_forecasts.index)
        n_assets = len(columns)

        lambdas, alphas, thetas = self.lambdas, self.alphas, self.thetas
        precision_matrices = self.precision_matrices
        stock_names = self.stock_names
        forecasts = self.model_forecasts
        
        if model.type == "regression":
            assert(precision_matrices is not None)
            weights = np.zeros(shape = (len(lambdas), len(alphas), len(thetas), n_dates, n_assets)) # lambdas x alphas x thetas x dates x assets
            for t, date in enumerate(forecasts.index):
                forecasts_date = forecasts.loc[date]
                prec_matrix_date = precision_matrices[date]
                for i,lam in enumerate(lambdas):
                    for j, _ in enumerate(alphas):
                        for k, _ in enumerate(thetas):
                            weight_t = pd.Series((1/lam)*(prec_matrix_date[j][k] @ forecasts_date), index = stock_names)
                            weight_t["IRX"] = 1 - weight_t.sum()
                            weights[i][j][k][t] = weight_t.values
            model_weights = (weights, forecasts.index, columns)

        elif model.type == "classification":
            weights = np.zeros(shape = (n_dates, n_assets)) # dates x assets
            for t, date in enumerate(forecasts.index):
                forecasts_date = forecasts.loc[date]
                n_pos = np.sum(forecasts_date > 0)
                n_neg = np.sum(forecasts_date < 0)
                weights_t = pd.Series(0., index = stock_names)
                weights_t.loc[(forecasts_date>0)] = 1/n_pos
                weights_t.loc[(forecasts_date<0)] = -1/n_neg
                weights_t["IRX"] = 1 - weights.sum()
                weights[t] = weights_t.values
            model_weights = (weights, forecasts.index, columns)
                
        return model_weights
        

