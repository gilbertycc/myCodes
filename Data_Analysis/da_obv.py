import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the ticker symbol and date range
ticker_symbol = input ("Enter ticker name: ")
data_period = '1y'

# Download the stock data from Yahoo Finance
ticker_data = yf.download(ticker_symbol, period=data_period, interval="1d")

# Compute the daily price change
price_change = ticker_data['Close'] - ticker_data['Close'].shift(1)

# Compute the direction of the price change
price_direction = np.where(price_change > 0, 1, -1)

# Compute the OBV
obv = price_direction * ticker_data['Volume']
obv = obv.cumsum()

# Compute the 21-day moving average of the OBV
obv_ma = obv.rolling(window=21).mean()

# Create a signal when the OBV crosses above its 21-day moving average
buy_signal = (obv > obv_ma) & (obv.shift(1) < obv_ma.shift(1))

# Create a signal when the OBV crosses below its 21-day moving average
sell_signal = (obv < obv_ma) & (obv.shift(1) > obv_ma.shift(1))

# Combine the buy and sell signals into a single signal
signal = pd.Series(0, index=ticker_data.index)
signal[buy_signal] = 1
signal[sell_signal] = -1

# Compute the daily returns of the stock
daily_returns = ticker_data['Close'].pct_change()

# Compute the strategy returns
strategy_returns = signal.shift(1) * daily_returns

# Compute the cumulative returns of the strategy
cumulative_strategy_returns = (strategy_returns + 1).cumprod()

# Compute the cumulative returns of the buy-and-hold strategy
cumulative_buy_and_hold_returns = (daily_returns + 1).cumprod()

# Plot the cumulative returns of the two strategies
plt.plot(cumulative_strategy_returns, label='OBV Strategy')
plt.plot(cumulative_buy_and_hold_returns, label='Buy and Hold')
plt.legend()
plt.show()
