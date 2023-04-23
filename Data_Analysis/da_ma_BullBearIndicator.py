import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class BullBearIndicator:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.stock_data = yf.download(ticker, start=start_date, end=end_date)
        self.stock_data = self.stock_data.dropna()
    
    def calculate_moving_averages(self, short_window, long_window):
        self.stock_data['SMA_short'] = self.stock_data['Adj Close'].rolling(window=short_window).mean()
        self.stock_data['SMA_long'] = self.stock_data['Adj Close'].rolling(window=long_window).mean()
    
    def plot_price_and_moving_averages(self):
        plt.figure(figsize=(20, 10))
        plt.plot(self.stock_data['Adj Close'], label='Price')
        plt.plot(self.stock_data['SMA_short'], label='SMA_short')
        plt.plot(self.stock_data['SMA_long'], label='SMA_long')
        plt.legend(loc='upper left')
        plt.show()
    
    def is_bullish(self):
        last_price = self.stock_data['Adj Close'].iloc[-1]
        sma_short = self.stock_data['SMA_short'].iloc[-1]
        sma_long = self.stock_data['SMA_long'].iloc[-1]
        
        if sma_short > sma_long and last_price > sma_short:
            return True
        else:
            return False
    
    def is_bearish(self):
        last_price = self.stock_data['Adj Close'].iloc[-1]
        sma_short = self.stock_data['SMA_short'].iloc[-1]
        sma_long = self.stock_data['SMA_long'].iloc[-1]
        
        if sma_short < sma_long and last_price < sma_short:
            return True
        else:
            return False



indicator = BullBearIndicator('META', '2023-01-01', '2023-04-22')
indicator.calculate_moving_averages(5, 14)

if indicator.is_bullish():
    print('Bullish signal detected')
elif indicator.is_bearish():
    print('Bearish signal detected')
else:
    print('No clear signal detected')

indicator.plot_price_and_moving_averages()

