# Quant Skill Building

Five Python projects I wrote to learn core quant ideas. These came out of my work in the Wall Street Quants course.

Each folder is one project with one script:

- 1-etf-momentum: Sharpe, drawdown, beta and alpha on a simple ETF momentum strategy
- 2-rolling-beta-alpha: rolling beta and alpha for tech stocks
- 3-combining-streams: combining two return streams optimally
- 4-zscore-momentum-crypto: z-score momentum on crypto
- 5-momentum-vs-reversal: momentum vs reversal across time horizons

Run any script locally:

    pip install -r requirements.txt
    python <folder>/<script>.py

Projects 1, 2 and 4 download prices from Yahoo Finance. Project 5 downloads prices from Binance. Project 3 uses simulated data.
