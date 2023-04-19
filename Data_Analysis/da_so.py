# Stochastic Oscillator (TA)


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define the ticker symbol and date range
ticker_symbol = input ("Enter ticker name: ")
data_period = '1y'

# Download the stock data from Yahoo Finance
ticker_data = yf.download(ticker_symbol, period=data_period)

# Calculate the Stochastic Oscillator
n = 14
high = ticker_data['High'].rolling(n).max()
low = ticker_data['Low'].rolling(n).min()
k = 100 * (ticker_data['Close'] - low) / (high - low)
d = k.rolling(3).mean()

# Plot the stock data and Stochastic Oscillator
plt.rcParams['figure.figsize'] = [12, 7]
plt.rc('font', size=14)
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(ticker_data['Close'])
ax[0].set_ylabel('Price')
ax[1].plot(k, label='%K(Main)')
ax[1].plot(d, label='%D(MA)')
ax[1].set_ylabel('Stochastic Oscillator')
ax[1].legend()
plt.show()

