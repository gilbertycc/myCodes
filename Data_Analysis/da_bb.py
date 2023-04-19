import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define the ticker symbol and date range
#ticker_symbol = "AAPL"
ticker_symbol = input ("Enter ticker name: ")
data_period = "1y"

# Download the stock data from Yahoo Finance
ticker_data = yf.download(ticker_symbol, period=data_period, interval="1d")

# Calculate the rolling mean and standard deviation
ticker_data['MA20'] = ticker_data['Close'].rolling(window=20).mean()
ticker_data['20dSTD'] = ticker_data['Close'].rolling(window=20).std()

# Calculate the upper and lower Bollinger Bands
ticker_data['UpperBand'] = ticker_data['MA20'] + (ticker_data['20dSTD'] * 2)
ticker_data['LowerBand'] = ticker_data['MA20'] - (ticker_data['20dSTD'] * 2)

# Plot the stock data and Bollinger Bands
plt.rcParams['figure.figsize'] = [12, 7]
plt.plot(ticker_data.index, ticker_data['Close'], label='Closing Price')
plt.plot(ticker_data.index, ticker_data['MA20'], label='20 Day Moving Average')
plt.plot(ticker_data.index, ticker_data['UpperBand'], label='Upper Bollinger Band')
plt.plot(ticker_data.index, ticker_data['LowerBand'], label='Lower Bollinger Band')
plt.legend(loc='upper left')
plt.title(ticker_symbol + ' Bollinger Bands')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
