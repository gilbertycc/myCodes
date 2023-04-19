import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Download data
ticker_symbol = 'AAPL'
#start_date = '2020-01-01'
#end_date = '2022-01-01'
data_period='max'
stock_data = yf.download(ticker_symbol, period=data_period)

# Extract the adjusted closing prices
closing_prices = stock_data['Close'].values

# Split data into training and testing sets
train_size = int(len(closing_prices) * 0.8)
x_train = np.arange(train_size).reshape(-1, 1)
y_train = closing_prices[:train_size]
x_test = np.arange(train_size, len(closing_prices)).reshape(-1, 1)
y_test = closing_prices[train_size:]

# Define the Gaussian Process Regressor model
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
gpr_model = GaussianProcessRegressor(kernel=kernel, random_state=0)

# Train the model
gpr_model.fit(x_train, y_train)

# Predict the stock prices using the test data
y_pred, sigma = gpr_model.predict(x_test, return_std=True)

# Compute performance metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Plot the predicted and actual closing prices
plt.plot(x_test, y_pred, label='Predicted')
plt.plot(x_test, y_test, label='Actual')
plt.fill_between(x_test.flatten(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.1)
plt.legend()
plt.xlabel('Days')
plt.ylabel('Closing Price')
plt.show()

# Print the performance metrics
print('R2 Score: {:.2f}'.format(r2))
print('Mean Squared Error: {:.2f}'.format(mse))
print('Root Mean Squared Error: {:.2f}'.format(np.sqrt(mse)))
print('Mean Absolute Error: {:.2f}'.format(mae))
