import yfinance as yf
import matplotlib.pyplot as plt



# ticker_name='NVDA'
ticker_name = input("Enter a ticker: ")

# download data for target symbol from Yahoo Finance
data = yf.download(ticker_name, period='1y')

# calculate the MA
data['MA5'] = data['Adj Close'].rolling(window=5).mean()
data['MA10'] = data['Adj Close'].rolling(window=10).mean()
data['MA50'] = data['Adj Close'].rolling(window=50).mean()
data['MA200'] = data['Adj Close'].rolling(window=200).mean()

# plot the data and the moving averages
plt.figure(figsize=(12,6))
plt.title('MA Chart of symbol: ' + ticker_name)
plt.plot(data.index, data['Adj Close'], label='Closing Price')
plt.plot(data.index, data['MA5'], label='MA5')
plt.plot(data.index, data['MA10'], label='MA10')
plt.plot(data.index, data['MA50'], label='MA50')
plt.plot(data.index, data['MA200'], label='MA200')
plt.legend(loc='upper left')
plt.show()