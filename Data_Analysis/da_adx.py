import yfinance as yf
import pandas as pd
import numpy as np

def calculate_directional_indicators(df):
    df['UpMove'] = df['High'].diff()
    df['DownMove'] = df['Low'].diff().abs()

    df['UpMove'] = np.where(df['UpMove'] > df['DownMove'], df['UpMove'], 0)
    df['DownMove'] = np.where(df['UpMove'] < df['DownMove'], df['DownMove'], 0)

    df['AvgUpMove'] = df['UpMove'].rolling(window=14).mean()
    df['AvgDownMove'] = df['DownMove'].rolling(window=14).mean()

    df['PosDI'] = df['AvgUpMove'] / df['TrueRange']
    df['NegDI'] = df['AvgDownMove'] / df['TrueRange']

    df['ADX'] = (100 * (df['PosDI'] - df['NegDI']).abs() / (df['PosDI'] + df['NegDI'])).rolling(window=14).mean()

    return df[['PosDI', 'NegDI', 'ADX']]

def calculate_true_range(df):
    df['TR1'] = abs(df['High'] - df['Low'])
    df['TR2'] = abs(df['High'] - df['Close'].shift())
    df['TR3'] = abs(df['Low'] - df['Close'].shift())

    df['TrueRange'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)

    return df.drop(['TR1', 'TR2', 'TR3'], axis=1)

# Define the ticker symbol and date range
ticker_symbol = input ("Enter ticker name: ")

data_period='1y'

# Download the stock data from Yahoo Finance
ticker_data = yf.download(ticker_symbol, period=data_period, interval="1d")

# Calculate the ADX
ticker_data = calculate_true_range(ticker_data)
ticker_data = calculate_directional_indicators(ticker_data)

# Plot the ADX
ticker_data[['ADX']].plot(figsize=(16,8))
