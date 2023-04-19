# Import required libraries
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Define the ticker symbol and date range
ticker_symbol = input("Enter a ticker name: ")
data_period = input("Enter a data period (1d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max): ")

# Download the stock data from Yahoo Finance
ticker_data = yf.download(ticker_symbol, period=data_period, interval="1d")

# Calculate the highest and lowest swing
highest_swing = -1
lowest_swing = -1

for i in range(1,ticker_data.shape[0]-1):
    if ticker_data['High'][i] > ticker_data['High'][i-1] and ticker_data['High'][i] > ticker_data['High'][i+1] and (highest_swing == -1 or ticker_data['High'][i] > ticker_data['High'][highest_swing]):
        highest_swing = i
    if ticker_data['Low'][i] < ticker_data['Low'][i-1] and ticker_data['Low'][i] < ticker_data['Low'][i+1] and (lowest_swing == -1 or ticker_data['Low'][i] < ticker_data['Low'][lowest_swing]):
        lowest_swing = i

# Define the ratios, colors and levels for Fibonacci retracements
ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
colors = ["black", "r", "g", "b", "cyan", "magenta", "yellow"]
levels = []
max_level = ticker_data['High'][highest_swing]
min_level = ticker_data['Low'][lowest_swing]

for ratio in ratios:
    if highest_swing > lowest_swing: # Uptrend
        levels.append(max_level - (max_level - min_level) * ratio)
    else: # Downtrend
        levels.append(min_level + (max_level - min_level) * ratio)

# Plot the stock data and Fibonacci retracements
plt.rcParams['figure.figsize'] = [12, 7]
plt.rc('font', size=14)
plt.plot(ticker_data['Close'])
start_date = ticker_data.index[min(highest_swing, lowest_swing)]
end_date = ticker_data.index[max(highest_swing, lowest_swing)]

for i in range(len(levels)):
    plt.hlines(levels[i], xmin=ticker_data.index[0], xmax=ticker_data.index[-1], label="{:.1f}%".format(ratios[i] * 100), colors=colors[i], linestyles="dashed")

plt.legend()

# Add a title to the plot
plt.title("{} Stock Data ({}) with Fibonacci Retracement Levels".format(ticker_symbol.upper(), data_period))

plt.show()
