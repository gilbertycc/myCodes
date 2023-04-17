'''
The code retrieves the historical stock price data of a given ticker symbol from Yahoo Finance, 
calculates the moving average, scales the data, trains an SVR model, makes predictions, and plots 
the actual and predicted values. The plot includes the ticker symbol, moving average length, 
and SVR kernel type.

'''

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime



def get_current_stockprice(ticker_name):

    ticker = yf.Ticker(ticker_name) 

    if "navPrice" in ticker.info:
      current_price = ticker.info["navPrice"]
    elif "currentPrice" in ticker.info:
      current_price = ticker.info["currentPrice"]
    else:
      current_price = ticker.info["ask"]

    return (current_price)



def stock_MA_data(symbol_Ticker, ma_days=30, period_length="max"):
    symbolEq = yf.Ticker(symbol_Ticker)
    symbolEq_history = symbolEq.history(period=period_length)

    # Calculate the moving averages of the 'Close' column
    moving_average = symbolEq_history['Close'].rolling(window=ma_days).mean()

    # Drop null values
    moving_average = moving_average.dropna()

    # Get the corresponding dates
    dates = symbolEq_history.loc[moving_average.index, :].index

    return moving_average, dates



# Plot the results
def plot_prediction_char(tk_name,mv_len,dates_df,actual_mv,pd_mv,svr_model):

    plt.title('MA' + str(mv_len) +' Analysis with SVR Model(' + svr_model +') on '+ tk_name)
    plt.plot(dates_df, actual_mv, label='Actual')
    plt.plot(dates_df[-len(pd_mv):], pd_mv, label='Predicted')
    plt.legend()
    plt.show()

def pf_review(date_data, mv_data, pd_data, print_table=False):

    # Crate a df Table view of the results
    df_result_view = pd.concat([mv_data.tail(test_data_days-1), pd.DataFrame(pd_data, index=date_data[-len(pd_data):], columns=['Predicted Close'])], axis=1)
    df_result_view.index = df_result_view.index.strftime('%Y-%m-%d')

    # Get the actual and predicted values
    y_true = df_result_view['Close']
    y_pred = df_result_view['Predicted Close']

    # Calculate the evaluation metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Print the evaluation metrics
    print("[*****Model Performnce Summary*****]")
    print("Mean Absolute Error: ", round(mae, 2))
    print("Mean Squared Error: ", round(mse, 2))
    print("Root Mean Squared Error: ", round(rmse, 2))
    print("R-squared: ", round(r2, 2))


    if print_table == True:
      print(df_result_view)


def pd_nth_days_result(date_data, scaled_last_n_days):
    # Make predictions on the scaled data
    predicted_last_n_days = svr.predict(scaled_last_n_days[:-1])

    # Inverse transform the predicted data using the same scaler object
    predicted_last_n_days = scaler.inverse_transform(predicted_last_n_days.reshape(-1, 1))

    # Get the last n dates from date_data
    last_n_dates = date_data[-len(predicted_last_n_days):]

    # Create a DataFrame with predicted prices and dates as index
    predicted_data = pd.DataFrame(predicted_last_n_days, index=last_n_dates, columns=["Predicted Close (n+1)"])
    predicted_data.index = predicted_data.index.strftime('%Y-%m-%d')

    # Print the DataFrame
    print(predicted_data)



#################### MAIN ####################

# variable setup
test_data_days = 10
mv_len=5
period_len='1y' #'1y,1w,max
# ticker_name = 'MSFT'
svr_model='linear' #'linear','poly','rbf'


# Ask for a ticker from stdin
ticker_name = input("Enter a symbol ticker: ")

# Extract the moving average values
mv_data, date_data = stock_MA_data(ticker_name, mv_len, period_len)

# Scale the data
scaler = StandardScaler()
scaled_data = (scaler.fit_transform(mv_data.values.reshape(-1, 1)))

# Split the data into training and testing sets
train_data = scaled_data[:-test_data_days]
test_data = scaled_data[-test_data_days:]

# Train the SVR model
'''
Linear Kernel: 'linear'
Polynomial Kernel: 'poly'
Radial Basis Function Kernel: 'rbf'
Sigmoid Kernel: 'sigmoid'
'''
svr = SVR(kernel=svr_model, C=1e3, gamma=0.1, epsilon=0.1)
svr.fit(train_data[:-1], train_data[1:].ravel())

# Make predictions on the test data
pd_data = svr.predict(test_data[:-1])

# Inverse transform the data
pd_data = scaler.inverse_transform(pd_data.reshape(-1, 1))

# Plot the results
plot_prediction_char(ticker_name,mv_len,date_data,mv_data,pd_data,svr_model)

# Performance Review
pf_review(date_data, mv_data, pd_data, False)


##### Performance prediction after trained the model #####
# Extract the last n days of data from the historical data
n_days = 5
last_n_days = mv_data[-n_days:].values.reshape(-1, 1)


# Append the current price to the nparry
current_price = np.array([ get_current_stockprice(ticker_name)]) 
last_n_days = np.append(last_n_days, [current_price], axis=0)

# Append the current date to date_data
date_data = date_data.tz_convert('UTC')
current_date = pd.Timestamp.now().strftime('%Y-%m-%d')
current_date_index = pd.DatetimeIndex([current_date], tz='UTC')

# append the new Index object to the existing Index
date_data = date_data.append(current_date_index)

# Scale the data using the same scaler object
scaled_last_n_days = scaler.transform(last_n_days)
print("Current price("+ticker_name+"):", current_price)
pd_nth_days_result(date_data, scaled_last_n_days)


