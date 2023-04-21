from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.covariance import graphical_lasso

class PrecisionMatrixBuilder:

    def __init__(self, returns, window, alphas = [0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 5], thetas = [0.55, 0.65, 0.75]):
        """Class to compute the precision matrix for a given set of returns and a given set of parameters

        Args:
            returns (_type_): _description_
            window (_type_): _description_
            alphas (list, optional): _description_. Defaults to [0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 5].
            thetas (list, optional): _description_. Defaults to [0.55, 0.65, 0.75].
        """
        self.returns = returns
        self.window = window
        self.alphas = alphas
        self.thetas = thetas

    def get_precision_matrix(self):
        """Covariance and precision matrix computation based on rolling windows and graphical lasso

        Args:
            returns (pd.DataFrame): sample returns
            window (int, optional): size of the rolling window. Defaults to 512.
            alphas (list[float], optional): penalization parameters for graphical lasso.

        Returns:
            dict: (dates, len(alphas) x 2 x returns.shape[1] x returns.shape[1] arrays)
            list: list of alphas used in GLASSO
            list: list of thetas used in EPO
        """
        cov = self.returns.ewm(min_periods = 1, alpha = 1/self.window).cov()
        dates = cov.index.levels[0]
        precision_matrix = {}
        for date in tqdm(dates):
            cov_date = cov.loc[date].values
            date_res = []
            if not np.isnan(cov_date).any():
                for alpha in self.alphas:
                    alpha_res = []
                    for theta in self.thetas:
                        cov_date_theta = (1-theta)*cov_date + theta*np.eye(cov_date.shape[0])
                        prec_date = graphical_lasso(cov_date_theta, alpha = alpha)[1]
                        alpha_res.append(prec_date)
                    date_res.append(alpha_res)
                precision_matrix[date] = date_res
            else:
                precision_matrix[date] = np.zeros((len(self.alphas), len(self.thetas), self.returns.shape[1], self.returns.shape[1]))
        return precision_matrix
