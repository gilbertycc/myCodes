import yfinance as yf
import pandas as pd
import numpy as np

# Define the ticker symbol and data period
ticker_symbol = input ('Enter a ticker name: ')
data_period = "1y"

# Download the stock data from Yahoo Finance
stock_data = yf.download(ticker_symbol, period=data_period)

# Define the time period for RSI calculation
rsi_period = 14

# Calculate the price change for each day
price_change = stock_data['Adj Close'].diff()

# Define the up days and down days
up_days = price_change.where(price_change > 0, 0)
down_days = -1 * price_change.where(price_change < 0, 0)

# Calculate the moving average of up days and down days
up_ma = up_days.rolling(window=rsi_period).mean()
down_ma = down_days.rolling(window=rsi_period).mean()

# Calculate the Relative Strength (RS)
rs = up_ma / down_ma

# Calculate the Relative Strength Index (RSI)
rsi = 100 - (100 / (1 + rs))

# Add RSI to the stock data
stock_data['RSI'] = rsi

# Plot the stock price and RSI
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
ax1.plot(stock_data['Adj Close'], label='Close')
ax2 = ax1.twinx()
ax2.plot(stock_data['RSI'], color='orange', label='RSI')
plt.title('Stock Price and RSI: '+ticker_symbol)
plt.legend(loc='upper left')
plt.show()
