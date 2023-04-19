import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt



ticker_symbol = input("Enter a ticker name: ")
data_period = "1y"

stock_data = yf.download(ticker_symbol, period=data_period)

# Calculate the 12-day and 26-day exponential moving averages (EMA):

ema_12 = stock_data['Adj Close'].ewm(span=12, adjust=False).mean()
ema_26 = stock_data['Adj Close'].ewm(span=26, adjust=False).mean()

#Calculate the MACD line by subtracting the 26-day EMA from the 12-day EMA:
macd = ema_12 - ema_26

# Calculate the signal line by taking the 9-day EMA of the MACD line:
signal_line = macd.ewm(span=9, adjust=False).mean()

# Plot the MACD line and the signal line:
plt.figure(figsize=(12,6))
plt.plot(stock_data.index, macd, label='MACD', color='blue')
plt.plot(stock_data.index, signal_line, label='Signal Line', color='red')
#plt.plot(stock_data.index, stock_data['Adj Close'], label='Closing Price')

plt.legend(loc='upper left')
plt.show()
