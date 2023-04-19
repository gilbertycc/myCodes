import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#ticker_symbol = 'MSFT'
ticker_symbol = input ('Enter a ticker name: ')
period_date='5y'
#start_date = '2010-01-01'
#end_date = '2021-12-31'
#stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)
stock_data = yf.download(ticker_symbol, period=period_date)


# Calculate the average daily return and daily standard deviation of the returns:
daily_returns = stock_data['Adj Close'].pct_change()
avg_daily_return = daily_returns.mean()
daily_std_dev = daily_returns.std()

# Calculate monthly returns
monthly_returns = stock_data['Adj Close'].resample('M').ffill().pct_change()

# Calculate annualized mean return and volatility
annualized_mean_return = np.mean(monthly_returns) * 12
annualized_volatility = np.std(monthly_returns) * np.sqrt(12)

# Calculate the risk-free rate, assuming 3% per annum
risk_free_rate = 0.03 / 12

# Calculate the monthly Sharpe Ratio
sharpe_ratio = (annualized_mean_return - risk_free_rate) / annualized_volatility


'''
# Calculate the annualized average return and annualized standard deviation:
annual_avg_return = avg_daily_return * 252
annual_std_dev = daily_std_dev * np.sqrt(252)

# Calculate the Sharpe Ratio:
sharpe_ratio = annual_avg_return / annual_std_dev
'''

# Calculate the rolling monthly Sharpe Ratio using a 12-month rolling window
rolling_sharpe_ratio = monthly_returns.rolling(window=12).apply(lambda x: ((np.mean(x) - risk_free_rate) / np.std(x)) * np.sqrt(12))

# Plot the rolling monthly Sharpe Ratio
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(rolling_sharpe_ratio.index, rolling_sharpe_ratio.values, color='blue')
ax.set(title='Rolling Monthly Sharpe Ratio', xlabel='Date', ylabel='Sharpe Ratio')
plt.show()


# Plot the daily returns:
daily_returns.plot(figsize=(10, 6))
plt.title("Daily Returns")
plt.xlabel("Date")
plt.ylabel("Returns")
plt.show()


# Plot the monthly returns:
monthly_returns.plot(figsize=(10, 6))
plt.title("Monthly Returns")
plt.xlabel("Date")
plt.ylabel("Returns")
plt.show()

'''
# Calculate and plot the rolling Sharpe Ratio:
rolling_window = 252
rolling_avg_return = daily_returns.rolling(window=rolling_window).mean()
rolling_std_dev = daily_returns.rolling(window=rolling_window).std()
rolling_sharpe_ratio = (rolling_avg_return * 252) / (rolling_std_dev * np.sqrt(252))

rolling_sharpe_ratio.plot(figsize=(10, 6))
plt.title("Rolling Sharpe Ratio")
plt.xlabel("Date")
plt.ylabel("Sharpe Ratio")
plt.show()
'''

# Calculate and print other performance indicators, such as the total return, cumulative return, and maximum drawdown:
total_return = stock_data['Adj Close'][-1] / stock_data['Adj Close'][0] - 1
cumulative_return = (1 + daily_returns).cumprod()[-1] - 1

cumulative_returns = (1 + daily_returns).cumprod()
rolling_max = cumulative_returns.rolling(window=252).max()
daily_drawdown = cumulative_returns / rolling_max - 1
max_drawdown = daily_drawdown.min()


#print("Sharpe Ratio:", sharpe_ratio)
print("Monthly Sharpe Ratio:", sharpe_ratio)
print("Total Return:", total_return)
print("Cumulative Return:", cumulative_return)
print("Max Drawdown:", max_drawdown)

