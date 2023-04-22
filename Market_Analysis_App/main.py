from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np
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
        plt.legend(loc='upper left')
        plt.show()


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
        plt.axhline(y=vwap_data[-1], color='black', linestyle='--')
        plt.axhline(y=25, color='gray', linestyle='--')
        plt.legend(loc='upper left')
        plt.show()


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



class Analysis_TA(BollingerBands, MovingAverage, ADX, VWAP, StochasticOscillator, RSI, MADC, FibonacciRetracement):
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

    def plot_returns(self):
        cumulative_returns = (self.stock_data / self.stock_data.iloc[0] - 1) * 100
        plt.figure(figsize=(12, 6))
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


def main():

    ##### Main #####

    review_stock = Analysis_TA("BA", '1y')
    print(review_stock)

    ''' Chart Analysis '''
    # review_stock.plot_chart_MovingAverage()
    # review_stock.plot_chart_BollingerBands()
    # review_stock.plot_chart_ADX()
    # review_stock.plot_chart_vwap()
    # review_stock.plot_chart_stochastic_oscillator()
    # review_stock.plot_chart_rsi()
    # review_stock.plot_chart_madc()
    # review_stock.plot_chart_fibonacci_retracement()

    ''' Signal - Bullish/Bearish
    Moving Averages: 
    Relative Strength Index (RSI)
    MACD (Moving Average Convergence Divergence)
    Bollinger Bands
    '''



    ''' Signal - oversold/overbought
    Relative Strength Index (RSI)
    Stochastic Oscillator
    Commodity Channel Index (CCI)
    Moving Average Convergence Divergence (MACD)
    '''

    ''' RR / SharpRatio '''
    sr = SharpeRatio("NVDA", '6mo', 0.04)
    sr.get_metrics()
    sr.plot_returns()

if __name__ == "__main__":
    main()
