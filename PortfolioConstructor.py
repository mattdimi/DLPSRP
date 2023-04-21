import numpy as np
import pandas as pd
from tqdm import tqdm

class PortfolioConstructor:

    def __init__(self, models, stock_names, model_forecasts, precision_matrices, alphas, thetas, lambdas = [0.5, 1, 2]):
        self.models = models
        self.stock_names = stock_names
        self.model_forecasts = model_forecasts
        self.precision_matrices = precision_matrices
        self.alphas = alphas
        self.thetas = thetas
        self.lambdas = lambdas
    
    def get_portfolios(self):
        
        model_weights = {}

        columns = self.stock_names.to_list() + ["IRX"]
        
        n_dates = len(self.model_forecasts[self.models[0].name].index)
        n_assets = len(columns)

        lambdas, alphas, thetas = self.lambdas, self.alphas, self.thetas
        precision_matrices = self.precision_matrices
        stock_names = self.stock_names
        
        for model in tqdm(self.models):

            forecasts = self.model_forecasts[model.name]

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
                model_weights[model.name] = (weights, forecasts.index, columns)

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
                model_weights[model.name] = (weights, forecasts.index, columns)
                
        return model_weights
        

