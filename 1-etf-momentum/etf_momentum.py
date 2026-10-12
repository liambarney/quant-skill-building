"""
Project 1: simple momentum strategy on four ETFs (QQQ, TLT, GLD, RWO).

Rule: hold an ETF when its trailing 252-day return per unit of risk is above 1,
equal-weight whatever passes, trade on the prior day's signal.
Reports Sharpe, yearly Sharpe, drawdown, beta and alpha vs SPY, and a cost estimate.

Run locally:  pip install yfinance pandas numpy matplotlib
Then:         python etf_momentum.py
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

data = yf.download(["QQQ", "TLT", "GLD", "RWO"], start="2016-01-01", end="2026-01-01", auto_adjust=True)["Close"]

def compute_returns(data):
    returns = data.pct_change().dropna()
    return returns
returns = compute_returns(data)
          
def compute_momentum(returns):
    momentum = (returns.rolling(252).mean() * 252) / (returns.rolling(252).std() * np.sqrt(252))
    return momentum
momentum = compute_momentum(returns)

def compute_portfolio (momentum_df):
    portfolio = (momentum_df > 1)*1
    portfolio = portfolio.div(portfolio.abs().sum(axis=1), axis=0)
    return portfolio
portfolio = compute_portfolio(momentum)

def portfolio_returns(portfolio_df, returns_df):
    strat_ret = (portfolio_df.shift() * returns_df).sum(axis=1)
    return strat_ret

strat_ret = portfolio_returns(portfolio, returns)
strat_ret = strat_ret.loc[momentum.dropna().index[0]:]

# 1 - annualized sharpe, whole period
print(strat_ret.mean() / strat_ret.std() * np.sqrt(252))

# 2 - sharpe within each year
def sharpe(x):
    return x.mean() / x.std() * np.sqrt(252)

print(strat_ret.groupby(strat_ret.index.year).apply(sharpe))

# 3 - correlation with underlying tickers
print(returns.corrwith(strat_ret))

# 4 - cumulative returns plot
strat_ret.cumsum().plot()
plt.show()

compounded = (1 + strat_ret).cumprod()
print(compounded.iloc[-1])
print(compounded.iloc[-1] ** (1/9) - 1)

# 5 - compounded curve + drawdown
compounded.plot(title="growth of $1")
plt.show()

drawdown = compounded / compounded.cummax() - 1
drawdown.plot(title="drawdown")
plt.show()
print("max drawdown:", drawdown.min())

# 6 - SPY benchmark comparison
spy = yf.download("SPY", start="2016-01-01", end="2026-01-01", auto_adjust=True)["Close"]
spy_ret = spy.pct_change().dropna()
spy_ret = spy_ret.loc[strat_ret.index[0]:]
print("SPY sharpe:", (spy_ret.mean() / spy_ret.std() * np.sqrt(252)).iloc[0])
print("strategy sharpe:", strat_ret.mean() / strat_ret.std() * np.sqrt(252))

spy_compounded = (1 + spy_ret).cumprod()
print("SPY growth multiple:", spy_compounded.iloc[-1].iloc[0])
print("SPY CAGR:", spy_compounded.iloc[-1].iloc[0] ** (1/9) - 1)

spy_series = spy_ret.iloc[:, 0]
beta = strat_ret.cov(spy_series) / spy_series.var()
print("strategy beta vs SPY:", beta)

# 7 - alpha
strat_ann = strat_ret.mean() * 252
spy_ann = spy_series.mean() * 252
alpha = strat_ann - beta * spy_ann
print("annualized alpha:", alpha)

# 8 - turnover and cost estimate
trades = portfolio.shift().fillna(0).diff().abs().sum(axis=1)
print("avg daily turnover:", trades.mean())

cost_per_unit = 0.0005   # 5 bps per dollar traded, conservative for liquid ETFs
strat_ret_net = strat_ret - trades.shift() * cost_per_unit
print("net sharpe:", strat_ret_net.mean() / strat_ret_net.std() * np.sqrt(252))
print("net CAGR:", (1 + strat_ret_net).cumprod().iloc[-1] ** (1/9) - 1)



