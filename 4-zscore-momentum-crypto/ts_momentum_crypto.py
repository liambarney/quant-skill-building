"""
Time-series momentum on crypto (z-score signal, tanh position sizing)

Universe: BTC, ETH, ADA, BNB, SOL, ZEC
Results measured from 2023-01-01 to today (change START below to whatever you want)

Run locally:  pip install yfinance pandas numpy matplotlib
Then:         python ts_momentum_crypto.py
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- settings -----------------
UNIV = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'BNB-USD', 'SOL-USD', 'ZEC-USD']
START = '2023-01-01'          # results are measured from here
DATA_START = '2022-01-01'     # one extra year of history so the 365-day baseline is full by START
FAST, SLOW = 10, 365          # lookbacks (days)
# ---------------------------------------------

# 1) returns
data = yf.download(UNIV, DATA_START, auto_adjust=False)
px = data['Adj Close']
ret = px / px.shift() - 1

# 2) z-score signal: sqrt(10) * (avg 10d ret - avg 365d ret) / (365d daily stdev)
#    The signal is computed from DATA_START, then results are sliced to START onward.
sig = np.sqrt(FAST) * (ret.rolling(FAST, min_periods=1).mean()
                       - ret.rolling(SLOW, min_periods=10).mean())
sig = sig / ret.rolling(SLOW, min_periods=10).std()

# 3) tanh squash -> weights in (-1, 1)
port = np.tanh(sig)

# 4) strategy returns: yesterday's weight * today's return (no lookahead)
strat_ret = (port.shift() * ret).loc[START:]
ret = ret.loc[START:]

# 5) Sharpe (annualized, crypto trades 365 days)
sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(365)
print('\n=== Strategy Sharpe (annualized) ===')
print(sharpe.round(2))

print('\n=== Buy & hold Sharpe, same window (for comparison) ===')
print((ret.mean() / ret.std() * np.sqrt(365)).round(2))

# 6) correlation, beta, alpha vs the underlying coin
print('\n=== Strategy vs underlying coin ===')
rows = {}
for c in ret.columns:
    both = pd.concat([strat_ret[c], ret[c]], axis=1, keys=['strat', 'coin']).dropna()
    corr = both['strat'].corr(both['coin'])
    beta = both['strat'].cov(both['coin']) / both['coin'].var()
    alpha_daily = both['strat'].mean() - beta * both['coin'].mean()
    rows[c] = {'corr': round(corr, 2),
               'beta': round(beta, 2),
               'alpha_ann_%': round(alpha_daily * 365 * 100, 1)}
print(pd.DataFrame(rows).T)

# plots
strat_ret.cumsum().plot(title='TS momentum: cumulative returns (sum of daily rets)')
plt.tight_layout(); plt.savefig('ts_momentum_cumret.png', dpi=120)
ret.cumsum().plot(title='Buy & hold: cumulative returns')
plt.tight_layout(); plt.savefig('buyhold_cumret.png', dpi=120)
print('\nSaved plots: ts_momentum_cumret.png, buyhold_cumret.png')