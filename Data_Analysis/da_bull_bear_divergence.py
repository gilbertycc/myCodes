import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define the ticker symbol
# ticker = 'QQQ'
ticker = input("Enter a ticker name: ")

# Get the data for the stock
data = yf.download(ticker, period='6mo')

# Calculate the Relative Strength Index (RSI)
delta = data['Adj Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
rsi = 100.0 - (100.0 / (1.0 + rs))

# Calculate the Moving Average Convergence Divergence (MACD)
ema12 = data['Adj Close'].ewm(span=12, adjust=False).mean()
ema26 = data['Adj Close'].ewm(span=26, adjust=False).mean()
macd = ema12 - ema26
signal = macd.ewm(span=9, adjust=False).mean()

# Calculate the bullish and bearish divergences
price = data['Adj Close']
bullish_divergence = (price > price.shift(1)) & (rsi < rsi.shift(1))
bearish_divergence = (price < price.shift(1)) & (rsi > rsi.shift(1))

# Plot the stock price and the indicators
plt.figure(figsize=(16, 8))
plt.plot(data.index, data['Adj Close'], label='Price')
plt.plot(data.index, rsi, label='RSI')
plt.plot(data.index, macd, label='MACD')
plt.plot(data.index, signal, label='Signal')
plt.plot(data.index, bullish_divergence * data['Adj Close'], 'g^', label='Bullish Divergence')
plt.plot(data.index, bearish_divergence * data['Adj Close'], 'rv', label='Bearish Divergence')
plt.axhline(y=30, color='gray', linestyle='--')
plt.axhline(y=70, color='gray', linestyle='--')

plt.legend(loc='upper left')
plt.show()
