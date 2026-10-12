"""
Project 2: rolling beta and alpha for five tech stocks against QQQ.

Beta = how much a stock moves per 1% move in QQQ (252-day rolling window, lagged one day).
Alpha = the stock's daily return minus beta times the QQQ return (what the market does not explain).
Compares volatility, correlation and Sharpe vs information ratio, raw vs alpha.

Run locally:  pip install yfinance pandas numpy
Then:         python rolling_beta_alpha.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
pd.set_option('display.float_format', lambda x: f'{x:.6f}')

data = yf.download([ "META", "AAPL", "AMZN", "NFLX", "GOOGL", "QQQ" ], start="2016-01-01", end="2026-01-01", auto_adjust=True)["Close"]

def compute_returns(data):
    returns = data.pct_change().dropna()
    return returns
returns = compute_returns(data)

def rolling_beta(returns, window=252):
    cov = returns.rolling(window).cov(returns['QQQ'])
    var = returns['QQQ'].rolling(window).var()
    beta = cov.div(var, axis=0)
    return beta

# lag beta one day so each day's alpha only uses a beta that was known the day before
beta = rolling_beta(returns).shift(1)
print(beta.tail())

def compute_alpha(returns, beta):
    alpha = returns - beta.mul(returns['QQQ'], axis=0)
    return alpha

# QQQ against itself is always zero alpha, so leave it out
alpha = compute_alpha(returns, beta).drop(columns='QQQ').dropna()
# compare raw returns and alpha on the same dates (the first year is the beta warm-up)
returns = returns.loc[alpha.index]
print(alpha.tail())

print(returns.std())
print(alpha.std())

print(returns.corr())
print(alpha.corr())

sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
ir = (alpha.mean() / alpha.std()) * np.sqrt(252)
print(pd.DataFrame({'sharpe': sharpe, 'IR': ir}))