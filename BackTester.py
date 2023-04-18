import numpy as np
import utils
import pandas as pd

class BackTester:

    def __init__(self, weights, returns, capital):
        self.weights = weights
        self.returns = returns
        self.returns, self.weights = self.returns.align(self.weights, join = 'inner')
        self.capital = capital
    
    def backtest(self):
        weighted_returns = (self.returns*self.weights).dropna()
        weighted_returns = weighted_returns.sum(axis = 1)
        cum_returns = self.capital*(1 + weighted_returns).cumprod()
        return weighted_returns, cum_returns
    
    def backtest_statistics(self):

        weighted_rets, cum_rets = self.backtest()

        average_rets = weighted_rets.mean()*252
        std = weighted_rets.std()*np.sqrt(252)
        neg_std = weighted_rets[weighted_rets<0].std()*np.sqrt(252)
        
        sharpe = average_rets/std
        sortino = average_rets/neg_std

        dd = utils.drawdown(weighted_rets)
        
        d = {"Rets":average_rets, "Std":std, "Max DD":dd, "Sharpe":sharpe, "Sortino":sortino}

        cum_rets.plot()

        return pd.Series(d)
        



        