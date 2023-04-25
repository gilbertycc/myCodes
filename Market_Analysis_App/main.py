from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
from base64 import b64encode
from IPython.core.display import HTML




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

    def get_stock_info_html(self):
        html = f"<p>Symbol Type: {self.symbol_type}</p>"
        html += f"<p>Ticker: {self.name_ticker}</p>"
        html += f"<p>Current Data Period: {self.data_period}</p>"
        html += f"<p>Latest Price: {self.get_current_price()}</p>"
        return html

    def __str__(self):
        return f"[***** Symbol Info *****]\nSymbol Type: {self.symbol_type}\nTicker: {self.name_ticker}\nCurrent Data Period: {self.data_period}\nLatest Price: {self.get_current_price()}"





class MovingAverage(Stock):
    def __init__(self, name_ticker, data_period='1y'):
        Stock.__init__(self, name_ticker)
        self.data_period = data_period
        self.chart_figsize = (20, 12)


    def plot_chart_MovingAverage(self):
        data_history = self.get_history_data()

        if data_history is None:
            return None

        # calculate the MA
        data_history['MA5'] = data_history['Adj Close'].rolling(window=5).mean()
        data_history['MA10'] = data_history['Adj Close'].rolling(window=10).mean()
        data_history['MA50'] = data_history['Adj Close'].rolling(window=50).mean()
        data_history['MA200'] = data_history['Adj Close'].rolling(window=200).mean()

        # plot the data and the moving averages
        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] MA Chart of symbol: {self.name_ticker} (Period: {self.data_period})")
        plt.plot(data_history.index, data_history['Adj Close'], label='Closing Price')
        plt.plot(data_history.index, data_history['MA5'], label='MA5')
        plt.plot(data_history.index, data_history['MA10'], label='MA10')
        plt.plot(data_history.index, data_history['MA50'], label='MA50')
        plt.plot(data_history.index, data_history['MA200'], label='MA200')
        plt.legend(loc='upper left')
        plt.show()

    def plot_chart_MovingAverage_html(self):
        data_history = self.get_history_data()

        if data_history is None:
            return None

        # calculate the MA
        data_history['MA5'] = data_history['Adj Close'].rolling(window=5).mean()
        data_history['MA10'] = data_history['Adj Close'].rolling(window=10).mean()
        data_history['MA50'] = data_history['Adj Close'].rolling(window=50).mean()
        data_history['MA200'] = data_history['Adj Close'].rolling(window=200).mean()

        # plot the data and the moving averages
        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] MA Chart of symbol: {self.name_ticker} (Period: {self.data_period})")
        plt.plot(data_history.index, data_history['Adj Close'], label='Closing Price')
        plt.plot(data_history.index, data_history['MA5'], label='MA5')
        plt.plot(data_history.index, data_history['MA10'], label='MA10')
        plt.plot(data_history.index, data_history['MA50'], label='MA50')
        plt.plot(data_history.index, data_history['MA200'], label='MA200')
        plt.legend(loc='upper left')
        plt_buffer = BytesIO()
        plt.savefig(plt_buffer, format='png')
        plt_buffer.seek(0)
        plt_base64 = b64encode(plt_buffer.read()).decode('utf-8')
        plt_html = f"<img src='data:image/png;base64,{plt_base64}'>"
        plt.close()
        return plt_html


class BollingerBands(Stock):
    def __init__(self, name_ticker, data_period='1y'):
        Stock.__init__(self, name_ticker)
        self.data_period = data_period
        self.chart_figsize = (20, 12)

    def plot_chart_BollingerBands(self):
        data_history = self.get_history_data()
        data_history['MA20'] = data_history['Close'].rolling(window=20).mean()
        data_history['20dSTD'] = data_history['Close'].rolling(window=20).std()
        data_history['UpperBand'] = data_history['MA20'] + (data_history['20dSTD'] * 2)
        data_history['LowerBand'] = data_history['MA20'] - (data_history['20dSTD'] * 2)

        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] Bollinger Bands of symbol: {self.name_ticker} (Period: {self.data_period})")
        plt.plot(data_history.index, data_history['Close'], label='Closing Price')
        plt.plot(data_history.index, data_history['MA20'], label='20 Day Moving Average')
        plt.plot(data_history.index, data_history['UpperBand'], label='Upper Bollinger Band')
        plt.plot(data_history.index, data_history['LowerBand'], label='Lower Bollinger Band')
        plt.fill_between(data_history.index, data_history['UpperBand'], data_history['LowerBand'], alpha=0.1)
        plt.legend(loc='upper left')
        plt.show()


    def plot_chart_BollingerBands_html(self):
        data_history = self.get_history_data()
        data_history = self.get_history_data()
        data_history['MA20'] = data_history['Close'].rolling(window=20).mean()
        data_history['20dSTD'] = data_history['Close'].rolling(window=20).std()
        data_history['UpperBand'] = data_history['MA20'] + (data_history['20dSTD'] * 2)
        data_history['LowerBand'] = data_history['MA20'] - (data_history['20dSTD'] * 2)

        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] Bollinger Bands of symbol: {self.name_ticker} (Period: {self.data_period})")
        plt.plot(data_history.index, data_history['Close'], label='Closing Price')
        plt.plot(data_history.index, data_history['MA20'], label='20 Day Moving Average')
        plt.plot(data_history.index, data_history['UpperBand'], label='Upper Bollinger Band')
        plt.plot(data_history.index, data_history['LowerBand'], label='Lower Bollinger Band')
        plt.fill_between(data_history.index, data_history['UpperBand'], data_history['LowerBand'], alpha=0.1)
        plt.legend(loc='upper left')
        plt_buffer = BytesIO()
        plt.savefig(plt_buffer, format='png')
        plt_buffer.seek(0)
        plt_base64 = b64encode(plt_buffer.read()).decode('utf-8')
        plt_html = f"<img src='data:image/png;base64,{plt_base64}'>"
        plt.close()
        return plt_html


class ADX(Stock):
    def __init__(self, name_ticker, data_period='1y'):
        Stock.__init__(self, name_ticker)
        self.data_period = data_period
        self.chart_figsize = (20, 12)

    def calculate_directional_indicators(self, df):
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

    def calculate_true_range(self, df):
        df['TR1'] = abs(df['High'] - df['Low'])
        df['TR2'] = abs(df['High'] - df['Close'].shift())
        df['TR3'] = abs(df['Low'] - df['Close'].shift())

        df['TrueRange'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)

        return df.drop(['TR1', 'TR2', 'TR3'], axis=1)

    '''ADX (Average Directional Index) is a technical analysis indicator'''
    def plot_chart_ADX(self):
        data_history = self.get_history_data()
        data_history = self.calculate_true_range(data_history)
        data_history = self.calculate_directional_indicators(data_history)

        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] ADX of symbol: {self.name_ticker} (Period: {self.data_period})")
        plt.plot(data_history.index, data_history['ADX'], label='ADX')
        plt.legend(loc='upper left')

        # Add dashed line at 25
        plt.axhline(y=25, color='gray', linestyle='--')
        plt.show()

    def plot_chart_ADX_html(self):
        data_history = self.get_history_data()
        data_history = self.calculate_true_range(data_history)
        data_history = self.calculate_directional_indicators(data_history)

        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] ADX of symbol: {self.name_ticker} (Period: {self.data_period})")
        plt.plot(data_history.index, data_history['ADX'], label='ADX')
        plt.legend(loc='upper left')

        # Add dashed line at 25
        plt.axhline(y=25, color='gray', linestyle='--')
        plt_buffer = BytesIO()
        plt.savefig(plt_buffer, format='png')
        plt_buffer.seek(0)
        plt_base64 = b64encode(plt_buffer.read()).decode('utf-8')
        plt_html = f"<img src='data:image/png;base64,{plt_base64}'>"
        plt.close()
        return plt_html



class VWAP(Stock):
    def __init__(self, name_ticker, data_period='1y'):
        Stock.__init__(self, name_ticker)
        self.data_period = data_period
        self.chart_figsize = (20, 12)

    
    def calculate_vwap(self):
        ticker_data = self.get_history_data()
        tp = (ticker_data['High'] + ticker_data['Low'] + ticker_data['Close']) / 3
        ticker_data['TP'] = tp
        ticker_data['TradedValue'] = ticker_data['TP'] * ticker_data['Volume']
        ticker_data['CumulativeTradedValue'] = ticker_data['TradedValue'].cumsum()
        ticker_data['CumulativeVolume'] = ticker_data['Volume'].cumsum()
        ticker_data['VWAP'] = ticker_data['CumulativeTradedValue'] / ticker_data['CumulativeVolume']
        return ticker_data['VWAP']
    
    def plot_chart_vwap(self):
        vwap_data = self.calculate_vwap()
        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] VWAP of symbol: {self.name_ticker} (Period: {self.data_period})")
        plt.plot(vwap_data.index, self.get_history_data()['Close'], label='Closing Price')
        plt.plot(vwap_data.index, vwap_data, label='VWAP')
        #plt.axhline(y=vwap_data[-1], color='black', linestyle='--')
        #plt.axhline(y=25, color='gray', linestyle='--')
        plt.legend(loc='upper left')
        plt.show()

    def plot_chart_vwap_html(self):
        vwap_data = self.calculate_vwap()
        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] VWAP of symbol: {self.name_ticker} (Period: {self.data_period})")
        plt.plot(vwap_data.index, self.get_history_data()['Close'], label='Closing Price')
        plt.plot(vwap_data.index, vwap_data, label='VWAP')
        #plt.axhline(y=vwap_data[-1], color='black', linestyle='--')
        #plt.axhline(y=25, color='gray', linestyle='--')
        plt.legend(loc='upper left')
        plt_buffer = BytesIO()
        plt.savefig(plt_buffer, format='png')
        plt_buffer.seek(0)
        plt_base64 = b64encode(plt_buffer.read()).decode('utf-8')
        plt_html = f"<img src='data:image/png;base64,{plt_base64}'>"
        plt.close()
        return plt_html


class StochasticOscillator(Stock):
    def __init__(self, name_ticker, data_period='1y',timeframe=14):
        Stock.__init__(self, name_ticker)
        self.data_period = data_period
        self.timeframe = timeframe
        self.chart_figsize = (20, 12)


    def set_timeframe(self, timeframe):
        self.timeframe = timeframe

    def calculate_stochastic_oscillator(self):

        ticker_data = self.get_history_data()
        
        high = ticker_data['High'].rolling(self.timeframe).max()
        low = ticker_data['Low'].rolling(self.timeframe).min()
        k = 100 * (ticker_data['Close'] - low) / (high - low)
        d = k.rolling(3).mean()
        return k, d
        
    def plot_chart_stochastic_oscillator(self):
        k, d = self.calculate_stochastic_oscillator()
        ticker_data = self.get_history_data()

        # Plot the stock data and Stochastic Oscillator
        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] Stochastic Oscillato of symbol: {self.name_ticker} (Period: {self.data_period}, Timeframe: {self.timeframe}) days")
        plt.plot(k, label='%K(Main)')
        plt.plot(d, label='%D(MA)')
        plt.axhline(y=20, color='gray', linestyle='--')
        plt.axhline(y=80, color='gray', linestyle='--')
        plt.legend(loc='upper left')
        plt.show()

    def plot_chart_stochastic_oscillator_html(self):
        k, d = self.calculate_stochastic_oscillator()
        ticker_data = self.get_history_data()

        # Plot the stock data and Stochastic Oscillator
        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] Stochastic Oscillato of symbol: {self.name_ticker} (Period: {self.data_period}, Timeframe: {self.timeframe}) days")
        plt.plot(k, label='%K(Main)')
        plt.plot(d, label='%D(MA)')
        plt.axhline(y=20, color='gray', linestyle='--')
        plt.axhline(y=80, color='gray', linestyle='--')
        plt.legend(loc='upper left')
        plt_buffer = BytesIO()
        plt.savefig(plt_buffer, format='png')
        plt_buffer.seek(0)
        plt_base64 = b64encode(plt_buffer.read()).decode('utf-8')
        plt_html = f"<img src='data:image/png;base64,{plt_base64}'>"
        plt.close()
        return plt_html


class RSI(Stock):
    def __init__(self, name_ticker, data_period='1y', timeframe=14):
        Stock.__init__(self, name_ticker)
        self.data_period = data_period
        self.timeframe = timeframe
        self.chart_figsize = (20, 12)

    def set_timeframe(self, timeframe):
        self.timeframe = timeframe

    def calculate_rsi(self):
        data_history = self.get_history_data()
        delta = data_history['Adj Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ma_up = up.rolling(window=self.timeframe).mean()
        ma_down = down.rolling(window=self.timeframe).mean()
        rs = ma_up / ma_down
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def plot_chart_rsi(self):
        data_history = self.get_history_data()

        if data_history is None:
            return None

        rsi = self.calculate_rsi()

        # plot the data and RSI
        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] RSI Chart of symbol: {self.name_ticker} (Period: {self.data_period}, Timeframe: {self.timeframe}) days")
        plt.plot(data_history.index, data_history['Adj Close'], label='Closing Price')
        plt.plot(rsi.index, rsi, label='RSI')
        plt.axhline(y=30, color='gray', linestyle='--')
        plt.axhline(y=70, color='gray', linestyle='--')
        plt.legend(loc='upper left')
        plt.show()

    def plot_chart_rsi_html(self):
        data_history = self.get_history_data()

        if data_history is None:
            return None

        rsi = self.calculate_rsi()

        # plot the data and RSI
        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] RSI Chart of symbol: {self.name_ticker} (Period: {self.data_period}, Timeframe: {self.timeframe}) days")
        plt.plot(data_history.index, data_history['Adj Close'], label='Closing Price')
        plt.plot(rsi.index, rsi, label='RSI')
        plt.axhline(y=30, color='gray', linestyle='--')
        plt.axhline(y=70, color='gray', linestyle='--')
        plt.legend(loc='upper left')
        plt_buffer = BytesIO()
        plt.savefig(plt_buffer, format='png')
        plt_buffer.seek(0)
        plt_base64 = b64encode(plt_buffer.read()).decode('utf-8')
        plt_html = f"<img src='data:image/png;base64,{plt_base64}'>"
        plt.close()
        return plt_html


class MADC(Stock):
    def __init__(self, name_ticker, data_period='1y',short_ma=5,long_ma=20):
        Stock.__init__(self, name_ticker)
        self.data_period = data_period
        self.chart_figsize = (20, 12)
        self.short_ma = short_ma
        self.long_ma = long_ma

    def calculate_madc(self):
        data_history = self.get_history_data()

        if data_history is None:
            return None

        # calculate the short-term and long-term exponential moving averages
        short_ema = data_history['Adj Close'].ewm(span=self.short_ma, adjust=False).mean()
        long_ema = data_history['Adj Close'].ewm(span=self.long_ma, adjust=False).mean()

        # calculate the MACD line
        macd_line = short_ema - long_ema

        # calculate the signal line
        signal_line = macd_line.ewm(span=9, adjust=False).mean()

        # calculate the histogram
        ma_diff = macd_line - signal_line

        return macd_line, signal_line, ma_diff

    def plot_chart_madc(self):
        macd_line, signal_line, ma_diff = self.calculate_madc()

        # plot the MACD line and signal line
        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] MADC Chart of symbol: {self.name_ticker} (Period: {self.data_period})")
        plt.plot(macd_line, label='MACD Line')
        plt.plot(signal_line, label='Signal Line')

        # plot the ma_diff
        plt.bar(ma_diff.index, ma_diff, width=0.5, align='center', label=(f"MA{self.short_ma} - MA{self.long_ma}"), color='gray')

        plt.legend(loc='upper left')
        plt.show()

    def plot_chart_madc_html(self):
        macd_line, signal_line, ma_diff = self.calculate_madc()

        # plot the MACD line and signal line
        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] MADC Chart of symbol: {self.name_ticker} (Period: {self.data_period})")
        plt.plot(macd_line, label='MACD Line')
        plt.plot(signal_line, label='Signal Line')

        # plot the ma_diff
        plt.bar(ma_diff.index, ma_diff, width=0.5, align='center', label=(f"MA{self.short_ma} - MA{self.long_ma}"), color='gray')

        plt.legend(loc='upper left')
        plt_buffer = BytesIO()
        plt.savefig(plt_buffer, format='png')
        plt_buffer.seek(0)
        plt_base64 = b64encode(plt_buffer.read()).decode('utf-8')
        plt_html = f"<img src='data:image/png;base64,{plt_base64}'>"
        plt.close()
        return plt_html

class FibonacciRetracement(Stock):

    def __init__(self, name_ticker, data_period='1y'):
        Stock.__init__(self, name_ticker)
        self.data_period = data_period
        self.chart_figsize = (20, 12)

    def calculate_swing(self, ticker_data):
        highest_swing = -1
        lowest_swing = -1

        for i in range(1, ticker_data.shape[0]-1):
            if ticker_data['High'][i] > ticker_data['High'][i-1] and ticker_data['High'][i] > ticker_data['High'][i+1] and (highest_swing == -1 or ticker_data['High'][i] > ticker_data['High'][highest_swing]):
                highest_swing = i
            if ticker_data['Low'][i] < ticker_data['Low'][i-1] and ticker_data['Low'][i] < ticker_data['Low'][i+1] and (lowest_swing == -1 or ticker_data['Low'][i] < ticker_data['Low'][lowest_swing]):
                lowest_swing = i

        return highest_swing, lowest_swing

    def calculate_fibonacci_levels(self, ticker_data, highest_swing, lowest_swing):
        ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        colors = ["black", "r", "g", "b", "cyan", "magenta", "yellow"]
        levels = []
        max_level = ticker_data['High'][highest_swing]
        min_level = ticker_data['Low'][lowest_swing]

        for ratio in ratios:
            if highest_swing > lowest_swing: # Uptrend
                levels.append(max_level - (max_level - min_level) * ratio)
            else: # Downtrend
                levels.append(min_level + (max_level - min_level) * ratio)

        return levels, colors, ratios

    def plot_chart_fibonacci_retracement(self):
        ticker_data = self.get_history_data()
        highest_swing, lowest_swing = self.calculate_swing(ticker_data)
        levels, colors, ratios = self.calculate_fibonacci_levels(ticker_data, highest_swing, lowest_swing)

        plt.rcParams['figure.figsize'] = self.chart_figsize
        plt.rc('font', size=14)
        plt.plot(ticker_data['Close'])
        start_date = ticker_data.index[min(highest_swing, lowest_swing)]
        end_date = ticker_data.index[max(highest_swing, lowest_swing)]

        for i in range(len(levels)):
            plt.hlines(levels[i], xmin=ticker_data.index[0], xmax=ticker_data.index[-1], label="{:.1f}%".format(ratios[i] * 100), colors=colors[i], linestyles="dashed")

        plt.legend()
        plt.title("[DA] {} Stock Data ({}) with Fibonacci Retracement Levels".format(self.name_ticker.upper(), self.data_period))
        plt.show()

    def plot_chart_fibonacci_retracement_html(self):
        ticker_data = self.get_history_data()
        highest_swing, lowest_swing = self.calculate_swing(ticker_data)
        levels, colors, ratios = self.calculate_fibonacci_levels(ticker_data, highest_swing, lowest_swing)

        plt.rcParams['figure.figsize'] = self.chart_figsize
        plt.rc('font', size=14)
        plt.plot(ticker_data['Close'])
        start_date = ticker_data.index[min(highest_swing, lowest_swing)]
        end_date = ticker_data.index[max(highest_swing, lowest_swing)]

        for i in range(len(levels)):
            plt.hlines(levels[i], xmin=ticker_data.index[0], xmax=ticker_data.index[-1], label="{:.1f}%".format(ratios[i] * 100), colors=colors[i], linestyles="dashed")

        plt.legend()
        plt.title("[DA] {} Stock Data ({}) with Fibonacci Retracement Levels".format(self.name_ticker.upper(), self.data_period))
        plt_buffer = BytesIO()
        plt.savefig(plt_buffer, format='png')
        plt_buffer.seek(0)
        plt_base64 = b64encode(plt_buffer.read()).decode('utf-8')
        plt_html = f"<img src='data:image/png;base64,{plt_base64}'>"
        plt.close()
        return plt_html


class OBV(Stock):
    def __init__(self, name_ticker, data_period='1y'):
        Stock.__init__(self, name_ticker)
        self.data_period = data_period
        self.chart_figsize = (20, 12)

    def calculate_obv(self):
        ticker_data = self.get_history_data()
        obv = []
        prev_obv = 0
        for i in range(1, len(ticker_data)):
            if ticker_data['Close'][i] > ticker_data['Close'][i-1]:
                current_obv = prev_obv + ticker_data['Volume'][i]
            elif ticker_data['Close'][i] < ticker_data['Close'][i-1]:
                current_obv = prev_obv - ticker_data['Volume'][i]
            else:
                current_obv = prev_obv
            obv.append(current_obv)
            prev_obv = current_obv
        return pd.Series(obv, index=ticker_data.index[1:])

    def plot_chart_obv(self):
        obv_data = self.calculate_obv()
        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] OBV of symbol: {self.name_ticker} (Period: {self.data_period})")
        plt.plot(obv_data.index, obv_data, label='OBV')
        plt.axhline(y=0, color='black', linestyle='--')
        plt.legend(loc='upper left')
        plt.show()


    def plot_chart_obv_html(self):
        obv_data = self.calculate_obv()
        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] OBV of symbol: {self.name_ticker} (Period: {self.data_period})")
        plt.plot(obv_data.index, obv_data, label='OBV')
        plt.axhline(y=0, color='black', linestyle='--')
        plt.legend(loc='upper left')
        plt_buffer = BytesIO()
        plt.savefig(plt_buffer, format='png')
        plt_buffer.seek(0)
        plt_base64 = b64encode(plt_buffer.read()).decode('utf-8')
        plt_html = f"<img src='data:image/png;base64,{plt_base64}'>"
        plt.close()
        return plt_html

class AccumulationDistributionLine(Stock):
    
    def __init__(self, name_ticker, data_period='1y'):
        Stock.__init__(self, name_ticker)
        self.data_period = data_period
        
        self.chart_figsize = (20, 12)
    
    def calculate_adl(self):
        self.df = self.get_history_data()
        self.df['CMF Multiplier'] = ((self.df['Close'] - self.df['Low']) - (self.df['High'] - self.df['Close'])) / (self.df['High'] - self.df['Low'])
        self.df['CMF Volume'] = self.df['CMF Multiplier'] * self.df['Volume']
        self.df['ADL'] = self.df['CMF Volume'].cumsum()
        return self.df[['ADL']]

    def plot_chart_ADL(self):
        self.calculate_adl()
        fig, ax = plt.subplots(figsize=self.chart_figsize)
        ax.plot(self.df.index, self.df['ADL'], label='ADL')
        ax.set(title=f"[DA] ADL of symbol: {self.name_ticker} (Period: {self.data_period})")
        ax.legend(loc='upper left')
        plt.show()

    def plot_chart_ADL_html(self):
        self.calculate_adl()
        fig, ax = plt.subplots(figsize=self.chart_figsize)
        ax.plot(self.df.index, self.df['ADL'], label='ADL')
        ax.set(title=f"[DA] ADL of symbol: {self.name_ticker} (Period: {self.data_period})")
        ax.legend(loc='upper left')
        plt_buffer = BytesIO()
        plt.savefig(plt_buffer, format='png')
        plt_buffer.seek(0)
        plt_base64 = b64encode(plt_buffer.read()).decode('utf-8')
        plt_html = f"<img src='data:image/png;base64,{plt_base64}'>"
        plt.close()
        return plt_html

class Analysis_TA(BollingerBands, MovingAverage, ADX, VWAP, StochasticOscillator, RSI, MADC, FibonacciRetracement, OBV, AccumulationDistributionLine):
    def __init__(self, name_ticker, data_period='1y'):
        Stock.__init__(self, name_ticker)
        StochasticOscillator.__init__(self, name_ticker)
        RSI.__init__(self, name_ticker)
        MADC.__init__(self, name_ticker)
        self.data_period = data_period




class SharpeRatio(Stock):
    def __init__(self, name_ticker, data_period='1y', risk_free_rate=0.05):
        Stock.__init__(self, name_ticker)
        self.data_period = data_period
        self.risk_free_rate = risk_free_rate
        self.stock_data = self.get_history_data()['Adj Close']
        self.daily_returns = self.stock_data.pct_change().dropna()
        self.annual_returns = self.daily_returns.mean() * 252
        self.annual_volatility = self.daily_returns.std() * np.sqrt(252)
        self.sharpe_ratio = (self.annual_returns - self.risk_free_rate) / self.annual_volatility
        self.chart_figsize = (20, 12)

    def plot_returns(self):
        cumulative_returns = (self.stock_data / self.stock_data.iloc[0] - 1) * 100
        plt.figure(figsize=self.chart_figsize)
        plt.plot(cumulative_returns)
        plt.title(f"Cumulative Returns of {self.name_ticker.upper()} ({self.data_period})")
        plt.ylabel("Cumulative Returns (%)")
        plt.show()

    def get_metrics(self):
        total_return = self.stock_data.iloc[-1] / self.stock_data.iloc[0] - 1
        cumulative_return = (self.stock_data.iloc[-1] / self.stock_data.iloc[0] - 1) * 100
        max_drawdown = (self.stock_data / self.stock_data.cummax() - 1).min() * 100
        monthly_returns = self.stock_data.resample('M').ffill().pct_change().dropna()
        sharpe_ratio = (monthly_returns.mean() - self.risk_free_rate / 12) / monthly_returns.std() * np.sqrt(12)
        
        print(f"{self.name_ticker.upper()} ({self.data_period}) Monthly Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Total Return: {total_return:.2f}")
        print(f"Cumulative Return: {cumulative_return:.2f}%")
        print(f"Max Drawdown: {max_drawdown:.2f}%")



class RRnRRR(Stock):
    def __init__(self, name_ticker):
        Stock.__init__(self, name_ticker)
        self.data = self.get_history_data()
        
    def get_rr(self, entry_price, stop_loss):
        """
        Calculates the risk-reward ratio (RR) for a given stock.

        Parameters:
        entry_price (float): The entry price of the stock.
        stop_loss (float): The stop loss price for the stock.

        Returns:
        float: The risk-reward ratio.
        """
        reward = abs(self.data['Close'].iloc[-1] - entry_price)
        risk = abs(entry_price - stop_loss)
        
        if reward == 0 or risk == 0:
            return 0
        
        return reward / risk
    
    def get_rrr(self, entry_price, stop_loss, target_price):
        """
        Calculates the risk-reward ratio (RR) for a given stock.

        Parameters:
        entry_price (float): The entry price of the stock.
        stop_loss (float): The stop loss price for the stock.
        target_price (float): The target price for the stock.

        Returns:
        float: The risk-reward ratio.
        """
        reward = abs(target_price - entry_price)
        risk = abs(entry_price - stop_loss)
        
        if reward == 0 or risk == 0:
            return 0
        
        return reward / risk



class BullBearIndicator(Stock):

    def __init__(self, name_ticker, data_period='1y'):
        Stock.__init__(self, name_ticker)
        self.data_period = data_period
        self.stock_data = self.get_history_data().dropna()
        self.short_window = 20
        self.long_window = 50
        self.tf_rsi = 14
        self.std_dev = 2
        self.tf_bb = 20

    ''' Signal - MA Indicator '''
    def calculate_moving_averages(self):
        self.stock_data['SMA_short'] = self.stock_data['Adj Close'].rolling(window=self.short_window).mean()
        self.stock_data['SMA_long'] = self.stock_data['Adj Close'].rolling(window=self.long_window).mean()
        
    def is_bullish_ma(self):

        self.calculate_moving_averages()
        last_price = self.stock_data['Adj Close'].iloc[-1]
        sma_short = self.stock_data['SMA_short'].iloc[-1]
        sma_long = self.stock_data['SMA_long'].iloc[-1]
        
        if sma_short > sma_long and last_price > sma_short:
            return True
        else:
            return False
    
    def is_bearish_ma(self):
        self.calculate_moving_averages()
        last_price = self.stock_data['Adj Close'].iloc[-1]
        sma_short = self.stock_data['SMA_short'].iloc[-1]
        sma_long = self.stock_data['SMA_long'].iloc[-1]
        
        if sma_short < sma_long and last_price < sma_short:
            return True
        else:
            return False

    ''' Signal - Breakdown Indicator '''          
    def calculate_sma(self, period):
        return self.stock_data['Adj Close'].rolling(window=period).mean()
    
    def is_bullish_bd(self):
        sma_short = self.calculate_sma(self.short_window)
        sma_long = self.calculate_sma(self.long_window)
        if sma_short[-1] > sma_long[-1] and sma_short[-2] < sma_long[-2]:
            return True
        else:
            return False
    
    def is_bearish_bd(self):
        sma_short = self.calculate_sma(self.short_window)
        sma_long = self.calculate_sma(self.long_window)
        if sma_short[-1] < sma_long[-1] and sma_short[-2] > sma_long[-2]:
            return True
        else:
            return False
        

    ''' Signal - OversoldSignal (RSI) Indicator '''
    def calculate_rsi(self, timefram):
        delta = self.stock_data['Adj Close'].diff().dropna()
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        avg_gain = gains.rolling(timefram).mean().dropna()
        avg_loss = -losses.rolling(timefram).mean().dropna()
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi
    
    def is_bullish_rsi(self):
        rsi = self.calculate_rsi(self.tf_rsi)
        if rsi.iloc[-1] < 30 and rsi.iloc[-2] > 30:
            return True
        else:
            return False
    
    def is_bearish_rsi(self):
        rsi = self.calculate_rsi(self.tf_rsi)
        if rsi.iloc[-1] > 70 and rsi.iloc[-2] < 70:
            return True
        else:
            return False

    ''' Signal - BollingerBands Indicator '''
    def calculate_bollinger_bands(self):
        # Calculate rolling mean and standard deviation
        self.stock_data['MA'] = self.stock_data['Close'].rolling(window=self.tf_bb).mean()
        self.stock_data['STD'] = self.stock_data['Close'].rolling(window=self.tf_bb).std()

        # Calculate upper and lower bands
        self.stock_data['Upper Band'] = self.stock_data['MA'] + (self.stock_data['STD'] * self.std_dev)
        self.stock_data['Lower Band'] = self.stock_data['MA'] - (self.stock_data['STD'] * self.std_dev)

    def is_bullish_bollinger_bands(self):
        self.calculate_bollinger_bands()
        # Check if current price is below the lower band line
        if self.stock_data['Close'].iloc[-1] < self.stock_data['Lower Band'].iloc[-1]:
            return True
        else:
            return False

    def is_bearish_bollinger_bands(self):
        self.calculate_bollinger_bands()
        # Check if current price is above the upper band line
        if self.stock_data['Close'].iloc[-1] > self.stock_data['Upper Band'].iloc[-1]:
            return True
        else:
            return False



def check_bullish(name_ticker, no_signal=2, data_period='6mo'):
    bb_signal = BullBearIndicator(name_ticker, data_period)
    indicators = [
        {'name': 'Moving Average', 'is_bullish': bb_signal.is_bullish_ma()},
        {'name': 'Breakdown', 'is_bullish': bb_signal.is_bullish_bd()},
        {'name': 'OversoldSignal(RSI)', 'is_bullish': bb_signal.is_bullish_rsi()},
        {'name': 'Bollinger Bands', 'is_bullish': bb_signal.is_bullish_bollinger_bands()}
    ]
    tickers = [name_ticker for i in indicators if i['is_bullish']]
    return name_ticker if len(tickers) >= no_signal else None

def check_bearish(name_ticker, no_signal=2, data_period='6mo'):
    bb_signal = BullBearIndicator(name_ticker, data_period)
    indicators = [
        {'name': 'Moving Average', 'is_bearish': bb_signal.is_bearish_ma()},
        {'name': 'Breakdown', 'is_bearish': bb_signal.is_bearish_bd()},
        {'name': 'OversoldSignal(RSI)', 'is_bearish': bb_signal.is_bearish_rsi()},
        {'name': 'Bollinger Bands', 'is_bearish': bb_signal.is_bearish_bollinger_bands()}
    ]
    tickers = [name_ticker for i in indicators if i['is_bearish']]
    return name_ticker if len(tickers) >= no_signal else None

def scan_sp500_bb():
  
    lst_bullish = []
    lst_bearish = []

    # Scrape S&P 500 tickers from Wikipedia
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    dfs = pd.read_html(url, header=0)
    sp500_df = dfs[0]
    sp500_tickers = sp500_df['Symbol'].tolist()

    # Replace "." with "-"
    sp500_tickers = [ticker.replace(".", "-") for ticker in sp500_tickers]

    # Create Tickers object with all the tickers
    tickers = yf.Tickers(sp500_tickers)

    for ticker in sp500_tickers:
        if check_bullish(ticker) != None: lst_bullish.append(ticker)
        if check_bearish(ticker) != None: lst_bearish.append(ticker)

    #print (f'Bullish stock: {lst_bullish}')
    #print (f'Bearish stock: {lst_bearish}')
    return lst_bullish, lst_bearish



def main():

    ##### Main #####

    #lst_bull, lst_bear = scan_sp500_bb()
    #shorted_list_stock = lst_bull + lst_bear
    #print (shorted_list_stock)

    shorted_list_stock = ['GOOG']

    for stock in shorted_list_stock:
      review_stock = Analysis_TA(stock, '1y')
      print(review_stock)      

      ''' Chart Analysis '''
      #review_stock.plot_chart_MovingAverage()
      #review_stock.plot_chart_BollingerBands()
      #review_stock.plot_chart_ADX()
      #review_stock.plot_chart_vwap()
      #review_stock.plot_chart_stochastic_oscillator()
      #review_stock.plot_chart_rsi()
      #review_stock.plot_chart_madc()
      #review_stock.plot_chart_fibonacci_retracement()
      #review_stock.plot_chart_obv()
      # review_stock.plot_chart_ADL()

      print(review_stock.get_stock_info_html())
      print(review_stock.plot_chart_MovingAverage_html())
      print(review_stock.plot_chart_BollingerBands_html())
      print(review_stock.plot_chart_ADX_html())
      print(review_stock.plot_chart_vwap_html())
      print(review_stock.plot_chart_stochastic_oscillator_html())
      print(review_stock.plot_chart_rsi_html())
      print(review_stock.plot_chart_madc_html())
      print(review_stock.plot_chart_fibonacci_retracement_html())
      print(review_stock.plot_chart_obv_html())
      print(review_stock.plot_chart_ADL_html())


      ''' Signal - Bullish/Bearish 
      Moving Averages 
      Breakdown
      OversoldSignal - Relative Strength Index (RSI)
      Bollinger Bands
      
    
      GREEN = '\033[92m'
      RED = '\033[91m'
      ENDC = '\033[0m'


      bb_signal = BullBearIndicator(stock, '6mo')
      print("[BB Signal - MA] ",end='')
      signal = 'Bullish signal detected' if bb_signal.is_bullish_ma() else 'Bearish signal detected' if bb_signal.is_bearish_ma() else 'No clear signal detected'
      color = GREEN if bb_signal.is_bullish_ma() else RED if bb_signal.is_bearish_ma() else ENDC
      print(color + signal + ENDC)
      print("[BB Signal - Breakdown] ",end='')
      signal = 'Bullish signal detected' if bb_signal.is_bullish_bd() else 'Bearish signal detected' if bb_signal.is_bearish_bd() else 'No clear signal detected'
      color = GREEN if bb_signal.is_bullish_bd() else RED if bb_signal.is_bearish_bd() else ENDC
      print(color + signal + ENDC)
      print("[BB Signal - OversoldSignal(RSI)] ",end='')
      signal = 'Bullish signal detected' if bb_signal.is_bullish_rsi() else 'Bearish signal detected' if bb_signal.is_bearish_rsi() else 'No clear signal detected'
      color = GREEN if bb_signal.is_bullish_rsi() else RED if bb_signal.is_bearish_rsi() else ENDC
      print(color + signal + ENDC)
      print("[BB Signal - Bollinger Bands] ",end='')
      signal = 'Bullish signal detected' if bb_signal.is_bullish_bollinger_bands() else 'Bearish signal detected' if bb_signal.is_bearish_bollinger_bands() else 'No clear signal detected'
      color = GREEN if bb_signal.is_bullish_bollinger_bands() else RED if bb_signal.is_bearish_bollinger_bands() else ENDC
      print(color + signal + ENDC)
      
      
      Signal - Technical analysis
      Signal - Momentum
      '''

      ''' Signal - oversold/overbought
      Relative Strength Index (RSI)
      Stochastic Oscillator
      Commodity Channel Index (CCI)
      Moving Average Convergence Divergence (MACD)
      '''

      ''' RR & RRR/ SharpRatio '''
      sr = SharpeRatio(stock, '6mo', 0.04)
      sr.get_metrics()
      #sr.plot_returns()

      '''
      rr = RRnRRR('FFIV')
      # Entry Price, Stop Loss, Target Price
      print("RR:", rr.get_rr(134, 130))
      print("RRR:", rr.get_rrr(134, 130, 140))
      '''
      ''' Strategy & Back test
      OBV Strategy
      Buy & Hold
      '''





if __name__ == "__main__":
    main()
