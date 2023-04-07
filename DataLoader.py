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
    tickers = yf.download(list(tickers) + index, start = start, ignore_tz = True)
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

def load_full_dataset(start = "2009-06-01"):

    cac_40 = load_cac_40_data(start)
    cac_40 = cac_40.reorder_levels([1, 0], axis = 1)
    
    stocks = cac_40.columns.levels[0].drop("^FCHI")

    train_year = 2014

    vix = load_vix(start).rename("VIX")
    vix_scaler = StandardScaler().fit(vix.loc[vix.index.year<=train_year].values.reshape(-1, 1))
    vix = pd.Series(vix_scaler.transform(vix.values.reshape(-1, 1)).squeeze(), index = vix.index)

    eurusd = load_eur_usd_ex_rate(start).rename("EURUSD").dropna()
    eurusd_scaler = StandardScaler().fit(eurusd.loc[eurusd.index.year<=train_year].values.reshape(-1, 1))
    eurusd = pd.Series(eurusd_scaler.transform(eurusd.values.reshape(-1, 1)).squeeze(), index = eurusd.index)

    french_ur = load_french_ur().squeeze()
    french_ur_scaler = StandardScaler().fit(french_ur.loc[french_ur.index.year<=train_year].values.reshape(-1, 1))
    french_ur = pd.Series(french_ur_scaler.transform(french_ur.values.reshape(-1, 1)).squeeze(), index = french_ur.index)
    french_ur_aligned = pd.Series(np.nan, index = cac_40.index)
    french_ur_aligned.update(french_ur)
    french_ur_aligned = french_ur_aligned.ffill().rename("UR")

    french_csi = load_french_consumer_sentiment_index().squeeze()
    french_csi_scaler = StandardScaler().fit(french_csi.loc[french_csi.index.year<=train_year].values.reshape(-1, 1))
    french_csi = pd.Series(french_csi_scaler.transform(french_csi.values.reshape(-1, 1)).squeeze(), index = french_csi.index)
    french_csi_aligned = pd.Series(np.nan, index = cac_40.index)
    french_csi_aligned.update(french_csi)
    french_csi_aligned = french_csi_aligned.ffill().rename("CSI")
    
    common_index = cac_40.index.intersection(vix.index).intersection(eurusd.index).intersection(french_ur_aligned.index).intersection(french_csi_aligned.index)

    lags = [1, 5, 10, 15, 20, 25, 30]

    res = {}

    for stock in stocks:
        
        stock_data = cac_40[stock]

        stock_features = {}

        for lag in lags:
            stock_features[stock + f"_{lag}"] = np.log(stock_data['Adj Close']).diff(lag)/lag
        
        stock_features["SGN"] = np.sign(np.log(stock_data['Adj Close']).diff(1))

        stock_features["FCHI"] = np.log(cac_40['^FCHI', 'Adj Close']).diff()

        # ATR
        high_low = stock_data['High'] - stock_data['Low']
        high_close = np.abs(stock_data['High'] - stock_data['Adj Close'].shift())
        low_close = np.abs(stock_data['Low'] - stock_data['Adj Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean()
        atr_scaler = StandardScaler().fit(atr.loc[atr.index.year<=train_year].values.reshape(-1, 1))
        atr = pd.Series(atr_scaler.transform(atr.values.reshape(-1, 1)).squeeze(), index = atr.index)
        stock_features["ATR"] = atr

        # RSI
        delta = stock_data['Adj Close'].diff()
        up = delta.where(delta > 0, 0)
        down = -delta.where(delta < 0, 0)
        mean_up = up.rolling(14).mean()
        mean_down = down.rolling(14).mean()
        rs = mean_up/mean_down
        rsi = 100*(1 - 1/(1+rs))
        rsi_scaler = StandardScaler().fit(rsi.loc[rsi.index.year<=train_year].values.reshape(-1, 1))
        rsi = pd.Series(rsi_scaler.transform(rsi.values.reshape(-1, 1)).squeeze(), index = rsi.index)
        stock_features["RSI"] = rsi
       
        # MACD
        macd = stock_data['Adj Close'].ewm(span = 26, adjust = False).mean() - stock_data['Adj Close'].ewm(span = 12, adjust = False).mean()
        macd_scaler = StandardScaler().fit(macd.loc[macd.index.year<=train_year].values.reshape(-1, 1))
        macd = pd.Series(macd_scaler.transform(macd.values.reshape(-1, 1)).squeeze(), index = macd.index)
        stock_features['MACD'] = macd

        stock_features["VIX"] = vix
        stock_features["EURUSD"] = eurusd
        stock_features["UR"] = french_ur_aligned
        stock_features["CSI"] = french_csi_aligned

        stock_features = pd.concat(stock_features, axis = 1)

        res[stock] = stock_features.loc[common_index, :]

    res = pd.concat(res, axis = 1).dropna()
    return res