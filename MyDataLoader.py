from typing import Tuple
import pandas as pd
import numpy as np
import yfinance as yf
import os
from tqdm import tqdm
import datetime as dt
from scipy.ndimage.interpolation import shift

class MyDataLoader:
    """
    Data loader class for the project
    Handles all the data loading and preprocessing
    """
    def __init__(self) -> None:
        self.full_dataset = None
        self.risk_free_rate = None
        self.lags = [5, 10, 15, 20, 25, 30]

    def load_cac_40_data(self, start = "2009-06-01", end = "2023-03-30"):
        """Loads the cac40 stock data from yahoo finance

        Args:
            start (str, optional): start date in the format yyyy-mm-dd. Defaults to "2009-06-01".

        Returns:
            pd.DataFrame: pd.DataFrame containing the stock data
        """
        cac40_assets = pd.read_html('https://en.wikipedia.org/wiki/CAC_40')[4]
        tickers = cac40_assets.Ticker
        tickers = list(tickers)
        tickers.remove('STLAP.PA')
        tickers.remove('WLN.PA')
        tickers.remove('URW.PA')
        index = ["^FCHI"]
        risk_free = ["^IRX"]
        tickers = yf.download(list(tickers) + index + risk_free, start = start, end = end, ignore_tz = True)
        return tickers

    def load_vix(self, start = "2009-06-01", end = "2023-03-30"):
        """Loads VIX data from yahoo finance

        Args:
            start (str, optional): start date in the format yyyy-mm-dd. Defaults to "2009-06-01".

        Returns:
            pd.Series: pd.Series of the VIX prices
        """
        vix = yf.download("^VIX", start = start, end = end, ignore_tz = True)
        return vix['Adj Close']

    def load_eur_usd_ex_rate(self, start = "2009-06-01", end = "2023-03-30"):
        """Loads EUR/USD exchange rate from yahoo finance

        Args:
            start (str, optional): start date in the format yyyy-mm-dd. Defaults to "2009-06-01".

        Returns:
            pd.Series: pd.Series of the EUR/USD exchange rate
        """
        eur_usd = yf.download("EURUSD=X", start = start, end = end, ignore_tz = True)
        return eur_usd['Adj Close']

    def load_french_ur(self):
        """Loads the french unemployment rate from FRED

        Returns:
            pd.Series: pd.Series of the french unemployment rate
        """
        french_ur = pd.read_csv("data/LRHUTTTTFRM156S.csv", index_col = 0)
        french_ur.index = pd.to_datetime(french_ur.index)
        return french_ur

    def load_french_consumer_sentiment_index(self):
        """Loads the french consumer sentiment index from FRED

        Returns:
            pd.Series: pd.Series of the french consumer sentiment index
        """
        french_csi = pd.read_csv("data/CSCICP03FRM665S.csv", index_col = 0)
        french_csi.index = pd.to_datetime(french_csi.index)
        return french_csi


    def load_full_dataset(self, start = "2009-06-01", end = "2023-03-30") -> pd.DataFrame:
        """Returns a pd.DataFrame with the full data

        Args:
            start (str, optional): start date in the format yyyy-mm-dd. Defaults to "2009-06-01".

        Returns:
            pd.DataFrame: pd.DataFrame with multiindex columns containing the full data
        """
        
        if self.full_dataset is not None:
            return self.full_dataset
        
        cac_40 = self.load_cac_40_data(start, end)
        cac_40 = cac_40.reorder_levels([1, 0], axis = 1)

        cac_40_index = yf.download("^FCHI", start=start, end=end, ignore_tz=True)
        cac_40_index = np.log(cac_40_index["Adj Close"]).diff()
        cac_40_index = pd.Series(cac_40_index.squeeze(), index = cac_40_index.index)
        
        risk_free_rate = cac_40['^IRX', 'Adj Close'].rename("RFR")/(100*252)
        self.risk_free_rate = risk_free_rate
        
        stocks = cac_40.columns.levels[0].drop(["^FCHI", "^IRX"])

        vix = self.load_vix(start, end).rename("VIX")
        vix = pd.Series(vix.squeeze(), index = vix.index)

        eurusd = self.load_eur_usd_ex_rate(start, end).rename("EURUSD").dropna()
        eurusd = pd.Series(eurusd.squeeze(), index = eurusd.index)

        french_ur = self.load_french_ur().squeeze()
        french_ur = pd.Series(french_ur.squeeze(), index = french_ur.index)
        french_ur_aligned = pd.Series(np.nan, index = cac_40.index)
        french_ur_aligned.update(french_ur)
        french_ur_aligned = french_ur_aligned.ffill().rename("UR")

        french_csi = self.load_french_consumer_sentiment_index().squeeze()
        french_csi = pd.Series(french_csi.squeeze(), index = french_csi.index)
        french_csi_aligned = pd.Series(np.nan, index = cac_40.index)
        french_csi_aligned.update(french_csi)
        french_csi_aligned = french_csi_aligned.ffill().rename("CSI")
        
        common_index = cac_40.index.intersection(vix.index).intersection(eurusd.index).intersection(french_ur_aligned.index).intersection(french_csi_aligned.index).intersection(risk_free_rate.index).intersection(cac_40_index.index)

        lags = self.lags

        res = {}

        for stock in stocks:
            
            stock_data = cac_40[stock]

            stock_features = {}

            stock_features["R"] = np.log(stock_data['Adj Close']).diff(1) - risk_free_rate
            stock_features["SGN"] = (np.sign(stock_features['R']) + 1)/2 # 0/1 classification

            for lag in lags:
                stock_features[stock + f"_{lag}"] = np.log(stock_data['Adj Close']).diff(lag)/lag

            stock_features["FCHI"] = np.log(cac_40['^FCHI', 'Adj Close']).diff()

            # ATR
            high_low = stock_data['High'] - stock_data['Low']
            high_close = np.abs(stock_data['High'] - stock_data['Adj Close'].shift())
            low_close = np.abs(stock_data['Low'] - stock_data['Adj Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(14).mean()
            stock_features["ATR"] = atr

            # RSI
            delta = stock_data['Adj Close'].diff()
            up = delta.where(delta > 0, 0)
            down = -delta.where(delta < 0, 0)
            mean_up = up.rolling(14).mean()
            mean_down = down.rolling(14).mean()
            rs = mean_up/mean_down
            rsi = 100*(1 - 1/(1+rs))
            stock_features["RSI"] = rsi
        
            # MACD
            macd = stock_data['Adj Close'].ewm(span = 26, adjust = False).mean() - stock_data['Adj Close'].ewm(span = 12, adjust = False).mean()
            stock_features['MACD'] = macd

            stock_features["VIX"] = vix
            stock_features["EURUSD"] = eurusd
            stock_features["UR"] = french_ur_aligned
            stock_features["CSI"] = french_csi_aligned
            
            # lag the features by 1 day as we are building a model to predict the next day
            for feature in stock_features.keys():
                if feature != "SGN" and feature != "R":
                    stock_features[feature] = stock_features[feature].shift(1)

            stock_features = pd.concat(stock_features, axis = 1)

            res[stock] = stock_features.loc[common_index]

        res = pd.concat(res, axis=1).dropna(axis = 0, how = "all").ffill().dropna(axis = 0, how = "any")
        
        self.full_dataset = res

        return res
    
    def load_feature_names(self):
        """Returns the names of the features

        Returns:
            list: name of the features used for a single-stock
        """
        features_names = ["R", "SGN"] + [f"R_{lag}" for lag in self.lags] + ["FCHI", "ATR", "RSI", "MACD", "VIX", "EURUSD", "UR", "CSI"]
        return features_names

    def load_full_dataset_array(self, start = "2009-06-01"):
        """Returns a numpy array of shape (time, stocks, features) and the stock names and time index

        Args:
            start (str, optional): Start time. Defaults to "2009-06-01".

        Returns:
            np.array: np.array of shape (number of timesteps, stocks, features), list of stock names, np.array of time index
        """
        df = self.load_full_dataset(start)
        stock_names = df.columns.levels[0]
        time_index = df.index
        res = []
        for stock in stock_names:
            stock_data = df.loc[:, stock]
            res.append(stock_data.values)
        return np.array(res).transpose(1, 0, 2), stock_names, time_index

    def load_benchmark(self, date_index, start = "2009-06-01"):
        """Loads CAC40 index benchmark returns

        Args:
            date_index (list): date index to align the benchmark with
            start (str, optional): start date in the format yyyy-mm-dd. Defaults to "2009-06-01".

        Returns:
            pd.Series: pd.Series of the benchmark returns
        """
        cac40 = yf.download("^FCHI", start=start, ignore_tz=True)
        cac40 = np.log(cac40["Adj Close"]).diff()
        cac40 = pd.Series(cac40.squeeze(), index = cac40.index)
        cac40 = cac40.loc[date_index]
        self.cac40_index = cac40
        return cac40

    def load_risk_free_rate(self, date_index, start = "2009-06-01"):
        """Load the risk free rate (3-month treasury yield)

        Args:
            date_index (list): date index to align the risk free rate with
            start (str, optional): start date in the format yyyy-mm-dd. Defaults to "2009-06-01".

        Returns:
            pd.Series: pd.Series of the risk free rate
        """
        if self.risk_free_rate is not None:
            return self.risk_free_rate
        treasury_yied_3mo = yf.download("^IRX", start=start, ignore_tz=True)/(100*252)
        treasury_yied_3mo = treasury_yied_3mo["Adj Close"]
        treasury_yied_3mo = treasury_yied_3mo.loc[date_index]
        self.risk_free_rate = treasury_yied_3mo
        return treasury_yied_3mo
    
    def load_returns(self):
        """Loads the returns of the stocks considered

        Returns:
            pd.DataFrame: pd.DataFrame of the returns of the stocks considered
        """
        dataset, stock_names, time_index = self.load_full_dataset_array()
        n_stocks = dataset.shape[1]
        returns = np.zeros((len(time_index), n_stocks))
        for stock in range(n_stocks):
            data_stock = dataset[:, stock, :]
            returns_stock = data_stock[:, 0]
            returns_stock = shift(returns_stock, 1, cval=np.NaN) # shift returns by 1-day to avoid forward looking bias
            returns[:, stock] = returns_stock
        returns = pd.DataFrame(returns, index = time_index, columns = stock_names).dropna()
        return returns
    
    def load_returns_with_risk_free(self):
        """Loads the returns of the stocks considered with the risk free rate

        Returns:
            pd.DataFrame: pd.DataFrame of the returns of the stocks considered with the risk free rate
        """
        dataset, stock_names, time_index = self.load_full_dataset_array()
        n_stocks = dataset.shape[1]
        returns = np.zeros((len(time_index), n_stocks))
        for stock in range(n_stocks):
            data_stock = dataset[:, stock, :]
            returns_stock = data_stock[:, 0] # do not shift returns in this case for backtesting
            returns[:, stock] = returns_stock
        returns = pd.DataFrame(returns, index = time_index, columns = stock_names).dropna()
        risk_free_rate = self.load_risk_free_rate(returns.index)
        returns = pd.concat([returns, risk_free_rate], axis = 1)
        returns.columns = returns.columns[:-1].tolist() + ["IRX"]
        return returns