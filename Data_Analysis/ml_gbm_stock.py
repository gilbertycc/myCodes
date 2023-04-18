import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# Define the stock ticker symbol and time period
# ticker_symbol = "TSLA"
#target_date = '2023-04-17'


ticker_symbol = input("Enter a ticker name: ")
target_date = input("Enter a target date in YYYY-MM-DD: ")
data_period = "max"

# Get the stock data from Yahoo Finance
stock_data = yf.download(ticker_symbol, period=data_period)

# Define the feature columns and target variable
feature_cols = ["Open", "High", "Low", "Volume"]
target_col = "Close"

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    stock_data[feature_cols], stock_data[target_col], test_size=0.3, random_state=55
)

# Define the Gradient Boosting Regressor model
gbm_model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=5, random_state=55)

# Train the model on the training data
gbm_model.fit(x_train, y_train)

# Predict the stock prices using the test data
y_pred = gbm_model.predict(x_test)

'''
print (y_pred)
predicted_df = pd.DataFrame(y_pred, index=x_test.index, columns=['Predicted'])
print (predicted_df)
'''


# Evaluate the model using R^2, RMSE, MAE
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)

print("Gradient Boosting Regressor R^2 score:", r2)
print("Gradient Boosting Regressor RMSE:", rmse)
print("Gradient Boosting Regressor MAE:", mae)





# Load the data for the target date
# target_date = '2023-04-17'
date_obj = datetime.strptime(target_date, '%Y-%m-%d')
next_day = date_obj + timedelta(days=1)
n_stock_data = yf.download(ticker_symbol, start=target_date, end=next_day.strftime("%Y-%m-%d"))


# Preprocess the data
n_test_data = n_stock_data[feature_cols]

# Make the prediction using the trained GBM model
#test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))
n_pred = gbm_model.predict(n_test_data)



# Print the predicted stock price
print('Predicted Stock Price('+ str(next_day.strftime("%Y-%m-%d")) +'): ' + str(n_pred))


