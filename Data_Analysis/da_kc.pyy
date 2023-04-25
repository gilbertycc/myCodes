'''
Day Trade DL & DH analysis

'''

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




class Stock:

    ''' Class attribute '''
    symbol_type = "stock"
    default_data_period = '1y'


    ''' Instance attribute '''
    def __init__(self, name_ticker):
      self.name_ticker = name_ticker
      self.data_period = self.default_data_period


    ''' Class Funcction '''
    def set_data_period(self, data_period):
        self.data_period = data_period

    def get_current_price(self):
        try:
            ticker_data = yf.Ticker(self.name_ticker)
            if "navPrice" in ticker_data.info:
                price_quote = ticker_data.info["navPrice"]
            elif "currentPrice" in ticker_data.info:
                price_quote = ticker_data.info["currentPrice"]
            else:
                price_quote = ticker_data.info["ask"]
            return price_quote
        except:
            print("Error: Could not retrieve current price.")
            return None

    def get_price_target_date(self, target_date):
        try:
            date_obj = datetime.strptime(target_date, '%Y-%m-%d')
            target_date_plus_one = date_obj + timedelta(days=1)
            ticker_data = yf.download(self.name_ticker, start=target_date, end=target_date_plus_one, progress=False)
            return (target_date, ticker_data['Adj Close'].values[0])
        except:
            print(f"[Error] Could not retrieve price for target date: {target_date}.")
            return None

    def get_price_history(self):
        try:
            ticker_data = yf.download(self.name_ticker, period=self.data_period, progress=False)
            return ticker_data['Adj Close']
        except:
            print("[Error] Could not retrieve price history.")
            return None

    def get_history_data(self):
        try:
            data_history = yf.download(self.name_ticker, period=self.data_period, progress=False)
        except Exception as e:
            print("[Error] occurred while getting data from Yahoo Finance API.")
            print(f"Error details: {e}")
            return None
        else:
            if data_history.empty:
                print("[Error] No data found for the entered ticker symbol.")
                return None
            else:
                return data_history


    def __str__(self):
        return f"[***** Symbol Info *****]\nSymbol Type: {self.symbol_type}\nTicker: {self.name_ticker}\nCurrent Data Period: {self.data_period}\nLatest Price: {self.get_current_price()}"


class KeltnerChannels(Stock):
    def __init__(self, name_ticker, data_period='1y', window=20, multiplier=2):
        Stock.__init__(self, name_ticker)
        self.data_period = data_period
        self.window = window
        self.multiplier = multiplier
        self.df = self.get_history_data()
        self.chart_figsize = (20, 12)


    
    def calculate_keltner_channels(self):
        self.df["tp"] = (self.df["High"] + self.df["Low"] + self.df["Close"]) / 3

        # ATR = [(n-1) * ATR + TR] / n
        atr = []
        tr1 = self.df["High"] - self.df["Low"]
        tr2 = abs(self.df["High"] - self.df["Close"].shift())
        tr3 = abs(self.df["Low"] - self.df["Close"].shift())
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)

        # initialize ATR on day 1 with the True Range value
        atr.append(tr[0])

        # calculate ATR for subsequent days
        for i in range(1, len(tr)):
            atr.append(((self.window - 1) * atr[-1] + tr[i]) / self.window)


        atr_df = pd.DataFrame({'ATR': atr})
        
        
        keltner_channels = pd.DataFrame(self.df["tp"])
        keltner_channels = pd.concat([keltner_channels, atr_df.set_axis(keltner_channels.index)], axis=1)
        ema_df = (self.df["tp"].ewm(span=self.window, adjust=False).mean())
        keltner_channels = pd.concat([keltner_channels, ema_df.rename("EMA")], axis=1)

        # Upper Band = MA(Typical Price) + Multiplier * ATR
        # Lower Band = MA(Typical Price) - Multiplier * ATR
        keltner_channels["upper_band"] = keltner_channels["EMA"] + self.multiplier * keltner_channels['ATR']
        keltner_channels["lower_band"] = keltner_channels["EMA"] - self.multiplier * keltner_channels['ATR']

        return keltner_channels
        

    def plot_chart_keltner_channels(self):
        keltner_channels = self.calculate_keltner_channels()
        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] Keltner Channels of symbol: {self.name_ticker} (Period: {self.data_period})")
        plt.plot(self.df.index, self.df['Close'], label='Closing Price')
        #plt.plot(self.df.index, self.df['High'], label='High')
        #plt.plot(self.df.index, self.df['Low'], label='Low')
        #plt.plot(self.df.index, self.df['Open'], label='Open')
        plt.plot(keltner_channels.index, keltner_channels['EMA'], label='EMA')
        plt.plot(keltner_channels.index, keltner_channels['upper_band'], label='Upper Keltner Channels')
        plt.plot(keltner_channels.index, keltner_channels['lower_band'], label='Lower Keltner Channels')
        plt.fill_between(keltner_channels.index, keltner_channels['upper_band'], keltner_channels['lower_band'], alpha=0.1)

        plt.legend(loc='upper left')
        plt.show()



### MAIN ###
name_ticker='MSFT'
kc = KeltnerChannels(name_ticker, data_period='6mo', window=20)
#kc.calculate_keltner_channels()
kc.plot_chart_keltner_channels()


