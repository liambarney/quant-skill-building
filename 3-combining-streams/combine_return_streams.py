"""
Project 3: combining two return streams optimally.

Two simulated daily return streams (X and HEDGE), each 10% return and 10% volatility,
correlation about -0.5. Weights come from inverse(covariance) @ expected returns.
Part 4 sweeps the expected return of HEDGE from -10% to +10% and tracks the weights and Sharpe.

Run locally:  pip install pandas numpy
Then:         python combine_return_streams.py
"""

import numpy as np
import pandas as pd

def gen_strat_returns():
    np.random.seed(5)

    corr = [[1, -0.5],
            [-0.5, 1]]

    corr = np.array(corr)

    vols = np.diag(np.array([0.1, 0.1])) / np.sqrt(252)

    sigma = vols @ corr @ vols

    mu = np.array([0.1, 0.1]) / 252

    dates = pd.date_range('20100101', '20191231', freq='B')

    rets = np.random.multivariate_normal(mu, sigma, size=len(dates))
    rets = pd.DataFrame(rets, columns=['X', 'HEDGE'], index=dates)

    rets = rets/rets.std()*0.1/np.sqrt(252)
    rets = rets-rets.mean()
    rets = rets+mu
    return rets

rets = gen_strat_returns()

# part 2: basic stats
ann_ret = rets.mean() * 252
ann_vol = rets.std() * np.sqrt(252)
sharpe = ann_ret / ann_vol
correl = rets.corr()

print("--- PART 2 ---")
print("Annualized returns:")
print(ann_ret)
print()
print("Annualized vols:")
print(ann_vol)
print()
print("Sharpes:")
print(sharpe)
print()
print("Correlation:")
print(correl)
print()

# part 3: optimal combo
mu = rets.mean()
sigma = rets.cov()

def optimal_weights(sigma, mu):
    wgt = np.linalg.inv(sigma) @ mu
    wgt = wgt / np.abs(wgt).sum()
    return wgt

wgt = optimal_weights(sigma, mu)
combo = (rets*wgt).sum(axis=1)

print("--- PART 3 ---")
print("Optimal weights [X, HEDGE]:")
print(wgt)
print()
print("Combo Sharpe:")
print(combo.mean()/combo.std()*np.sqrt(252))
print()

# part 4: sweep HEDGE's return from -10% to +10%
get_sharpe = lambda x: x.mean()/x.std()*np.sqrt(252)

wgts = {}
sharpes = {}
for hedge_ret in np.arange(-0.1, 0.11, 0.01):

    rets['HEDGE'] = rets['HEDGE'] - rets['HEDGE'].mean() + hedge_ret / 252
    mu = rets.mean()

    wgt = pd.Series(optimal_weights(sigma, mu), mu.index)

    hedge_ret_str = f'{int(round(hedge_ret*100))}%'
    wgts[hedge_ret_str] = wgt

    sharpes[hedge_ret_str] = get_sharpe((rets*wgt).sum(axis=1))

wgts = pd.DataFrame(wgts).T
sharpes = pd.Series(sharpes)

print("--- PART 4 ---")
print("Optimal weights at each HEDGE return:")
print(wgts)
print()
print("Combo Sharpe at each HEDGE return:")
print(sharpes)