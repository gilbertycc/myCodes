import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf

# Define the ticker symbol and date range
ticker_symbol = "0005.HK"
data_period='1y'

# Download the stock data from Yahoo Finance
ticker_data = yf.download(ticker_symbol, period=data_period, interval="1d")

# Compute the VWAP
vwap = (ticker_data['Volume'] * (ticker_data['High'] + ticker_data['Low']) / 2).cumsum() / ticker_data['Volume'].cumsum()

# Plot the candlestick chart
mpf.plot(ticker_data, type='candle', mav=(20), volume=True, figratio=(16,8), figscale=1.5)

# Highlight the bullish engulfing pattern
for i in range(1, len(ticker_data)):
    if ticker_data['Close'][i-1] < ticker_data['Open'][i-1] and ticker_data['Close'][i] > ticker_data['Open'][i] and ticker_data['Open'][i] < ticker_data['Close'][i-1] and ticker_data['Close'][i] > ticker_data['Open'][i-1]:
        plt.annotate('Bullish Engulfing', xy=(i, ticker_data['Low'][i]), xytext=(i, ticker_data['Low'][i]-2), ha='center', va='bottom', fontsize=12, color='g', fontweight='bold')

# Highlight the bearish engulfing pattern
for i in range(1, len(ticker_data)):
    if ticker_data['Close'][i-1] > ticker_data['Open'][i-1] and ticker_data['Close'][i] < ticker_data['Open'][i] and ticker_data['Open'][i] > ticker_data['Close'][i-1] and ticker_data['Close'][i] < ticker_data['Open'][i-1]:
        plt.annotate('Bearish Engulfing', xy=(i, ticker_data['High'][i]), xytext=(i, ticker_data['High'][i]+2), ha='center', va='bottom', fontsize=12, color='r', fontweight='bold')

# Show the plot
plt.show()
