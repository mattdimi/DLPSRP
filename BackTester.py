import numpy as np
import utils
import pandas as pd
import matplotlib.pyplot as plt

class BackTester:

    def __init__(self, weights, returns, capital, risk_free_asset):
        """Class to run backtests on portfolios
        
        Args:
            weights (pd.DataFrame): _description_
            returns (pd.DataFrame): _description_
            capital (float): _description_
            risk_free_asset (pd.Series): _description_
        """
        self.weights = weights
        self.returns = returns
        self.returns, self.weights = self.returns.align(self.weights, join = 'inner')
        self.capital = capital
        self.risk_free_asset = risk_free_asset

    def run_backtest(self):
        turnover = self.weights.diff().abs().dropna().sum(axis = 1)
        weighted_returns = (self.returns*self.weights).dropna()
        weighted_returns = weighted_returns.sum(axis = 1)

        index_inter = weighted_returns.index.intersection(turnover.index).intersection(self.risk_free_asset.index)
        weighted_returns = weighted_returns.loc[index_inter]
        turnover = turnover.loc[index_inter]
        risk_free_asset = self.risk_free_asset.loc[index_inter]

        weighted_returns = weighted_returns - turnover*0.0033*2 - risk_free_asset # 0.33% transaction cost
        cum_returns = self.capital* (1 + weighted_returns.cumsum())
        return weighted_returns, cum_returns
    
    def get_backtest_statistics(self, plot_performance = False):

        weighted_rets, cum_rets = self.run_backtest()

        average_turnover = self.weights.diff().abs().dropna().sum(axis = 1).mean()

        average_rets = weighted_rets.mean()*252
        std = weighted_rets.std()*np.sqrt(252)
        neg_std = weighted_rets[weighted_rets<0].std()*np.sqrt(252)
        
        sharpe = average_rets/std
        sortino = average_rets/neg_std
        dd = utils.drawdown(weighted_rets)
        calmar = average_rets/dd

        backtest_statistics = {
            "Yearly returns":average_rets,
            "Yearly standard deviation":std,
            "Max DD":dd,
            "Yearly Sharpe":sharpe,
            "Yearly Calmar":calmar,
            "Yearly Sortino":sortino,
            "Daily Turnover": average_turnover
            }

        if plot_performance:
            cum_rets.plot()
        
        return pd.Series(backtest_statistics)

        



        