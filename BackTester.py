import numpy as np
import utils
import pandas as pd
import matplotlib.pyplot as plt

class BackTester:

    def __init__(self, weights, returns, capital, risk_free_asset, benchmark = None, name = None):
        """Class to run backtests on portfolios

        Args:
            weights (pd.DataFrame): portfolio weights timeseries
            returns (pd.DataFrame): returns timeseries
            capital (float): capital allocation
            risk_free_asset (pd.Series): returns series of the risk free asset
        """
        self.weights = weights
        self.returns = returns
        self.returns, self.weights = self.returns.align(self.weights, join = 'inner')
        self.capital = capital
        self.risk_free_asset = risk_free_asset
        self.benchmark = benchmark
        self.time_index = self.weights.index
        self.name = name if name is not None else "Strategy"

    def run_backtest(self):
        """Runs the backtest and returns the returns and cumulative returns of the strategy

        Returns:
            tuple: (pd.Series, pd.Series)
        """
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

    def get_strategy_cumulative_returns(self):
        """Returns the cumulative returns of the strategy vs. the benchmark

        Returns:
            pd.DataFrame: cumulative returns of the strategy vs. the benchmark
        """
        _, cum_returns = self.run_backtest()
        benchmark = self.benchmark
        benchmark = benchmark.loc[cum_returns.index]
        benchmark_cum_returns = self.capital*(1+benchmark.cumsum())
        return pd.concat({self.name:cum_returns, "CAC 40":benchmark_cum_returns}, axis = 1)

    def get_backtest_statistics(self):
        """Returns the backtest statistics of the strategy

        Returns:
            pd.Series: pd.Series containing the backtest statistics
        """
        
        weighted_rets, _ = self.run_backtest()

        average_turnover = self.weights.diff().abs().dropna().sum(axis = 1).mean()

        average_rets = weighted_rets.mean()*252
        std = weighted_rets.std()*np.sqrt(252)
        neg_std = weighted_rets[weighted_rets<0].std()*np.sqrt(252)
        
        sharpe = average_rets/std
        sortino = average_rets/neg_std
        dd = utils.drawdown(weighted_rets)
        calmar = average_rets/dd

        backtest_statistics = {
            "Yearly excess returns":average_rets,
            "Yearly standard deviation":std,
            "Max DD":dd,
            "Yearly Sharpe":sharpe,
            "Yearly Calmar":calmar,
            "Yearly Sortino":sortino,
            "Daily Turnover": average_turnover
            }

        return pd.Series(backtest_statistics)
            
    

        



        