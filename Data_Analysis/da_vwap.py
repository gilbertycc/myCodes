import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define the ticker symbol and date range
ticker_symbol = input ("Enter ticker name: ")
data_period='1y'

# Download the stock data from Yahoo Finance
ticker_data = yf.download(ticker_symbol, period=data_period, interval="1d")

# Define the VWAP function
def vwap(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP'] = tp
    df['TradedValue'] = df['TP'] * df['Volume']
    df['CumulativeTradedValue'] = df['TradedValue'].cumsum()
    df['CumulativeVolume'] = df['Volume'].cumsum()
    df['VWAP'] = df['CumulativeTradedValue'] / df['CumulativeVolume']
    return df['VWAP']

# Calculate the VWAP for the stock data
ticker_data['VWAP'] = vwap(ticker_data)

# Plot the stock data and VWAP
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(ticker_data['Close'], label='Close')
ax.plot(ticker_data['VWAP'], label='VWAP')
ax.legend()
ax.set(title='Stock Price with VWAP', ylabel='Price ($)')
plt.show()
