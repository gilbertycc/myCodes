import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt


# Define the stock ticker symbol and time period
# ticker_symbol = "TSLA"

ticker_symbol = input('Enter a ticker name: ')

data_period = "max"
look_back = 60

# Get the stock data from Yahoo Finance
stock_data = yf.download(ticker_symbol, period=data_period)

# Extract the closing price data
closing_prices = stock_data['Close'].values.reshape(-1, 1)


# Scale the data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
closing_prices_scaled = scaler.fit_transform(closing_prices)

# Split the data into training and testing sets
train_size = int(len(closing_prices_scaled) * 0.8)
test_size = len(closing_prices_scaled) - train_size
train_data = closing_prices_scaled[0:train_size, :]
test_data = closing_prices_scaled[train_size:len(closing_prices_scaled), :]



# Create the feature and target datasets
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset(train_data, look_back)
testX, testY = create_dataset(test_data, look_back)

# Reshape the data for LSTM input
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# Define the LSTM model architecture
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))



# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model on the training data
model.fit(trainX, trainY, epochs=100, batch_size=32)

# Predict the stock prices using the test data
predicted_stock_prices = model.predict(testX)
predicted_stock_prices = scaler.inverse_transform(predicted_stock_prices)

# Evaluate the model using RMSE
rmse = np.sqrt(np.mean(((predicted_stock_prices - scaler.inverse_transform(testY.reshape(-1,1))) ** 2)))
print('Root Mean Squared Error:', rmse)


# plot the data and the moving averages
plt.figure(figsize=(12,6))
plt.title('Performance Chart of LSTM model: ' + ticker_symbol)
plt.plot(stock_data.index, closing_prices, label='Closing Price')
plt.plot(stock_data.index[-len(predicted_stock_prices):], predicted_stock_prices, label='Predit Price')
plt.legend(loc='upper right')
plt.show()
