import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Define the stock ticker symbol and time period
#ticker_symbol = "0005.HK"
data_period='max'
ticker_symbol = input('Enter a ticker: ')



# Get the stock data from Yahoo Finance
stock_data = yf.download(ticker_symbol, period=data_period)


# Define the feature columns and target variable
feature_cols = ["Open", "High", "Low", "Volume"]
target_col = "Close"

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    stock_data[feature_cols], stock_data[target_col], test_size=0.4, random_state=55
)

# Define the Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=400, random_state=55)

# Train the model on the training data
rf_model.fit(x_train, y_train)

# Predict the stock prices using the test data
y_pred = rf_model.predict(x_test)


# Evaluate the model using R^2, RMSE, and MAPE
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print("Random Forest Regressor R^2 score:", r2)
print("Random Forest Regressor RMSE:", rmse)
print("Random Forest Regressor MAPE:", mape)



# predict today close 
current_data = yf.download(ticker_symbol, period='1d')
r_pred = rf_model.predict(current_data[feature_cols])
print ("[INFO] Prediction of " + ticker_symbol + " closing price: ")
print ("Date       Close")
print (current_data.index[-1].strftime('%Y-%m-%d'), round(r_pred[-1],2))

