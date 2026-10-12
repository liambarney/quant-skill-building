"""
Project 5: cross-sectional (XS) momentum vs reversal at 4h to 24h horizons.

Builds the rank-demeaned-normalized portfolio at 6 lookbacks, computes Sharpes
for (a) normal 1-bar lag and (b) skip-a-bar 2-bar lag, on a date window you set.

Run locally:  pip install python-binance pandas numpy matplotlib
Then:         python xs_momentum_reversal.py

NOTE on universe: MATIC was rebranded to POL, so on recent windows MATICUSDT may
return nothing. The script skips symbols with no data and tells you.
"""

from binance.client import Client as bnb_client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------- settings -----------------
UNIV = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'XRPUSDT', 'DOTUSDT', 'SOLUSDT']
START, END = '2023-01-01', '2026-09-01'
FREQ = '4h'
BARS_PER_YEAR = 365 * 24 / 4              # crypto trades every day
# ---------------------------------------------

client = bnb_client(tld='US')   # drop tld='US' if you're outside the US

def get_binance_px(symbol, freq, start_ts, end_ts):
    data = client.get_historical_klines(symbol, freq, start_ts, end_ts)
    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_volume', 'num_trades', 'taker_base_volume',
            'taker_quote_volume', 'ignore']
    data = pd.DataFrame(data, columns=cols)
    data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
    return data.set_index('open_time')['close']

px = {}
for x in UNIV:
    try:
        s = get_binance_px(x, FREQ, START, END)
        if len(s) == 0:
            print(f'!! {x}: no data returned, skipping')
            continue
        px[x] = s
    except Exception as e:
        print(f'!! {x}: failed ({e}), skipping')

px = pd.DataFrame(px).astype(float)
px = px.reindex(pd.date_range(px.index[0], px.index[-1], freq=FREQ))
ret = px.pct_change()
print(f'\nUniverse: {list(px.columns)}  |  {len(px)} bars  {px.index[0]} -> {px.index[-1]}')

def xs_strategy(ret, hor, lag):
    """Rank-demeaned-normalized XS momentum portfolio.
    hor = lookback in 4h bars; lag = 1 (normal) or 2 (skip most recent bar)."""
    w = ret.rolling(hor, min_periods=1).mean().rank(axis=1)   # rank across coins
    w = w.subtract(w.mean(axis=1), axis=0)                          # demean -> dollar neutral
    w = w.divide(w.abs().sum(axis=1), axis=0)                       # normalize -> gross = 1
    return (w.shift(lag) * ret).sum(axis=1)

horizons = [1, 2, 3, 4, 5, 6]   # x4h = 4,8,12,16,20,24 hour lookbacks

normal = pd.DataFrame({h * 4: xs_strategy(ret, h, lag=1) for h in horizons})
skip   = pd.DataFrame({h * 4: xs_strategy(ret, h, lag=2) for h in horizons})

sr_normal = normal.mean() / normal.std() * np.sqrt(BARS_PER_YEAR)
sr_skip   = skip.mean()   / skip.std()   * np.sqrt(BARS_PER_YEAR)

out = pd.DataFrame({'momentum_sharpe': sr_normal.round(2),
                    'skip_a_bar_sharpe': sr_skip.round(2)})
out.index.name = 'lookback_hours'
print('\n=== Annualized Sharpes by lookback ===')
print('(negative = momentum loses = reversal dominates at that horizon)')
print(out)

# per-year stability check (a strategy that only worked in one year is suspect)
yr = normal.resample('YE').apply(lambda x: x.mean() / x.std() * np.sqrt(BARS_PER_YEAR))
print('\n=== Momentum Sharpe by year ===')
print(yr.round(2))

out.plot(kind='bar', title=f'XS Sharpe by lookback, {START} to {END}')
plt.axhline(0, color='k', lw=0.5)
plt.tight_layout(); plt.savefig('xs_sharpes.png', dpi=120)
print('\nSaved plot: xs_sharpes.png')