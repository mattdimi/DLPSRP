from typing import Tuple
import pandas as pd
import numpy as np
import yfinance as yf
import os
from tqdm import tqdm
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_cac_40_data(start = "2009-06-01"):
    cac40_assets = pd.read_html('https://en.wikipedia.org/wiki/CAC_40')[4]
    tickers = cac40_assets.Ticker
    tickers = list(tickers)
    tickers.remove('STLAP.PA')
    tickers.remove('WLN.PA')
    index = ["^FCHI"]
    risk_free = ["^IRX"]
    tickers = yf.download(list(tickers) + index + risk_free, start = start, ignore_tz = True)
    return tickers

def load_vix(start = "2009-06-01"):
    vix = yf.download("^VIX", start = start, ignore_tz = True)
    return vix['Adj Close']

def load_eur_usd_ex_rate(start = "2009-06-01"):
    eur_usd = yf.download("EURUSD=X", start = start, ignore_tz = True)
    return eur_usd['Adj Close']

def load_french_ur():
    french_ur = pd.read_csv("data/LRHUTTTTFRM156S.csv", index_col = 0)
    french_ur.index = pd.to_datetime(french_ur.index)
    return french_ur

def load_french_consumer_sentiment_index():
    french_csi = pd.read_csv("data/CSCICP03FRM665S.csv", index_col = 0)
    french_csi.index = pd.to_datetime(french_csi.index)
    return french_csi

def load_full_dataset(start = "2009-06-01") -> pd.DataFrame:
    """Returns a pd.DataFrame with the full data

    Args:
        start (str, optional): _description_. Defaults to "2009-06-01".

    Returns:
        pd.DataFrame: _description_
    """

    cac_40 = load_cac_40_data(start)
    cac_40 = cac_40.reorder_levels([1, 0], axis = 1)
    
    risk_free_rate = cac_40['^IRX', 'Adj Close'].rename("RFR")/(100*252)
    
    stocks = cac_40.columns.levels[0].drop(["^FCHI", "^IRX"])

    vix = load_vix(start).rename("VIX")
    vix = pd.Series(vix.squeeze(), index = vix.index)

    eurusd = load_eur_usd_ex_rate(start).rename("EURUSD").dropna()
    eurusd = pd.Series(eurusd.squeeze(), index = eurusd.index)

    french_ur = load_french_ur().squeeze()
    french_ur = pd.Series(french_ur.squeeze(), index = french_ur.index)
    french_ur_aligned = pd.Series(np.nan, index = cac_40.index)
    french_ur_aligned.update(french_ur)
    french_ur_aligned = french_ur_aligned.ffill().rename("UR")

    french_csi = load_french_consumer_sentiment_index().squeeze()
    french_csi = pd.Series(french_csi.squeeze(), index = french_csi.index)
    french_csi_aligned = pd.Series(np.nan, index = cac_40.index)
    french_csi_aligned.update(french_csi)
    french_csi_aligned = french_csi_aligned.ffill().rename("CSI")
    
    common_index = cac_40.index.intersection(vix.index).intersection(eurusd.index).intersection(french_ur_aligned.index).intersection(french_csi_aligned.index).intersection(risk_free_rate.index)

    lags = [5, 10, 15, 20, 25, 30]

    res = {}

    for stock in stocks:
        
        stock_data = cac_40[stock]

        stock_features = {}

        stock_features["R"] = np.log(stock_data['Adj Close']).diff(1) - risk_free_rate
        stock_features["SGN"] = np.sign(stock_features['R'])

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

        res[stock] = stock_features.loc[common_index, :]

    res = pd.concat(res, axis = 1).dropna()
    return res

def load_full_dataset_array(start = "2009-06-01"):
    """Returns a numpy array of shape (time, stocks, features) and the stock names and time index

    Args:
        start (str, optional): Start time. Defaults to "2009-06-01".

    Returns:
        _type_: np.array of shape (number of timesteps, stocks, features), list of stock names, np.array of time index
    """
    df = load_full_dataset(start)
    stock_names = df.columns.levels[0]
    time_index = df.index
    res = []
    for stock in stock_names:
        stock_data = df.loc[:, stock]
        res.append(stock_data.values)
    return np.array(res).transpose(1, 0, 2), stock_names, time_index.to_numpy()