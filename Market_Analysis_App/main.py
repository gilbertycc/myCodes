from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
from base64 import b64encode
from retrying import retry
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Stock:
    symbol_type = "stock"
    default_data_period = '1y'

    def __init__(self, name_ticker):
        self.name_ticker = name_ticker
        self.data_period = self.default_data_period

    def set_data_period(self, data_period):
        self.data_period = data_period

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
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
        except Exception as e:
            logging.error(f"Could not retrieve current price for {self.name_ticker}: {e}")
            return None

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def get_price_target_date(self, target_date):
        try:
            date_obj = datetime.strptime(target_date, '%Y-%m-%d')
            target_date_plus_one = date_obj + timedelta(days=1)
            ticker_data = yf.download(self.name_ticker, start=target_date, end=target_date_plus_one, progress=False, auto_adjust=True)
            if ticker_data.empty:
                logging.error(f"No price data for {self.name_ticker} on {target_date}")
                return None
            return (target_date, ticker_data['Close'].values[0])
        except Exception as e:
            logging.error(f"Could not retrieve price for {self.name_ticker} on {target_date}: {e}")
            return None

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def get_price_history(self):
        try:
            ticker_data = yf.download(self.name_ticker, period=self.data_period, progress=False, auto_adjust=True)
            if ticker_data.empty:
                logging.error(f"No price history for {self.name_ticker}")
                return None
            return ticker_data['Close']
        except Exception as e:
            logging.error(f"Could not retrieve price history for {self.name_ticker}: {e}")
            return None

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def get_history_data(self):
        try:
            data_history = yf.download(self.name_ticker, period=self.data_period, progress=False, auto_adjust=True)
            if data_history.empty:
                logging.error(f"No data found for {self.name_ticker}")
                return None
            return data_history
        except Exception as e:
            logging.error(f"Error fetching data for {self.name_ticker}: {e}")
            return None

    def get_stock_info_html(self):
        current_price = self.get_current_price()
        price_str = f"{current_price:.2f}" if current_price else "N/A"
        html = f"<p>Symbol Type: {self.symbol_type}</p>"
        html += f"<p>Ticker: {self.name_ticker}</p>"
        html += f"<p>Current Data Period: {self.data_period}</p>"
        html += f"<p>Latest Price: {price_str}</p>"
        return html

    def ml_RFR_html(self):
        try:
            model_name = 'RandomForestRegressor'
            stock_data = yf.download(self.name_ticker, period=self.data_period, progress=False)
            if stock_data.empty:
                logging.error(f"No data for {self.name_ticker} in ml_RFR_html")
                return f"<tr><td>{self.name_ticker}</td><td>N/A</td><td>{model_name}</td><td>N/A</td><td>N/A</td></tr>"
            
            feature_cols = ["Open", "High", "Low", "Volume"]
            target_col = "Close"
            x_train, x_test, y_train, y_test = train_test_split(
                stock_data[feature_cols], stock_data[target_col], test_size=0.2, random_state=55
            )
            
            rf_model = RandomForestRegressor(n_estimators=400, random_state=55)
            rf_model.fit(x_train, y_train)
            y_pred = rf_model.predict(x_test)
            
            r2 = r2_score(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            
            current_data = yf.download(self.name_ticker, period='1d', progress=False)
            if current_data.empty:
                logging.error(f"No current data for {self.name_ticker}")
                return f"<tr><td>{self.name_ticker}</td><td>N/A</td><td>{model_name}</td><td>N/A</td><td>N/A</td></tr>"
            
            r_pred = rf_model.predict(current_data[feature_cols])
            html = f"<tr><td>{self.name_ticker}</td><td>{current_data.index[-1].strftime('%Y-%m-%d')}</td><td>{model_name}</td><td>{round(float(r_pred[-1]), 2)}</td><td>{round(float(current_data['Close'].values[-1]), 2)}</td></tr>"
            return html
        except Exception as e:
            logging.error(f"Error in ml_RFR_html for {self.name_ticker}: {e}")
            return f"<tr><td>{self.name_ticker}</td><td>N/A</td><td>{model_name}</td><td>N/A</td><td>N/A</td></tr>"

    def __str__(self):
        current_price = self.get_current_price()
        price_str = f"{current_price:.2f}" if current_price else "N/A"
        return f"[***** Symbol Info *****]\nSymbol Type: {self.symbol_type}\nTicker: {self.name_ticker}\nCurrent Data Period: {self.data_period}\nLatest Price: {price_str}"

class MovingAverage(Stock):
    def __init__(self, name_ticker, data_period='1y'):
        super().__init__(name_ticker)
        self.data_period = data_period
        self.chart_figsize = (20, 12)

    def plot_chart_MovingAverage_html(self):
        data_history = self.get_history_data()
        if data_history is None:
            logging.error(f"No data for MovingAverage plot for {self.name_ticker}")
            return "<p>No Moving Average chart available</p>"
        
        data_history['MA5'] = data_history['Close'].rolling(window=5).mean()
        data_history['MA10'] = data_history['Close'].rolling(window=10).mean()
        data_history['MA50'] = data_history['Close'].rolling(window=50).mean()
        data_history['MA200'] = data_history['Close'].rolling(window=200).mean()
        
        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] MA Chart of symbol: {self.name_ticker} (Period: {self.data_period})")
        plt.plot(data_history.index, data_history['Close'], label='Closing Price')
        plt.plot(data_history.index, data_history['MA5'], label='MA5')
        plt.plot(data_history.index, data_history['MA10'], label='MA10')
        plt.plot(data_history.index, data_history['MA50'], label='MA50')
        plt.plot(data_history.index, data_history['MA200'], label='MA200')
        plt.legend(loc='upper left')
        
        plt_buffer = BytesIO()
        plt.savefig(plt_buffer, format='png', bbox_inches='tight')
        plt_buffer.seek(0)
        plt_base64 = b64encode(plt_buffer.read()).decode('utf-8')
        plt_html = f"<img src='data:image/png;base64,{plt_base64}' class='chart-img' onclick='enlargeImage(this)' ondblclick='downsizeImage(this)'>"
        plt.close()
        return plt_html

class BollingerBands(Stock):
    def __init__(self, name_ticker, data_period='1y'):
        super().__init__(name_ticker)
        self.data_period = data_period
        self.chart_figsize = (20, 12)

    def plot_chart_BollingerBands_html(self):
        data_history = self.get_history_data()
        if data_history is None:
            logging.error(f"No data for BollingerBands plot for {self.name_ticker}")
            return "<p>No Bollinger Bands chart available</p>"
        
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
        plt.savefig(plt_buffer, format='png', bbox_inches='tight')
        plt_buffer.seek(0)
        plt_base64 = b64encode(plt_buffer.read()).decode('utf-8')
        plt_html = f"<img src='data:image/png;base64,{plt_base64}' class='chart-img' onclick='enlargeImage(this)' ondblclick='downsizeImage(this)'>"
        plt.close()
        return plt_html

class ADX(Stock):
    def __init__(self, name_ticker, data_period='1y'):
        super().__init__(name_ticker)
        self.data_period = data_period
        self.chart_figsize = (20, 12)

    def calculate_directional_indicators(self, df):
        try:
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
        except Exception as e:
            logging.error(f"Error calculating ADX for {self.name_ticker}: {e}")
            return None

    def calculate_true_range(self, df):
        try:
            df['TR1'] = abs(df['High'] - df['Low'])
            df['TR2'] = abs(df['High'] - df['Close'].shift())
            df['TR3'] = abs(df['Low'] - df['Close'].shift())
            df['TrueRange'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
            return df.drop(['TR1', 'TR2', 'TR3'], axis=1)
        except Exception as e:
            logging.error(f"Error calculating True Range for {self.name_ticker}: {e}")
            return None

    def plot_chart_ADX_html(self):
        data_history = self.get_history_data()
        if data_history is None:
            logging.error(f"No data for ADX plot for {self.name_ticker}")
            return "<p>No ADX chart available</p>"
        
        data_history = self.calculate_true_range(data_history)
        if data_history is None:
            return "<p>No ADX chart available</p>"
        
        data_history = self.calculate_directional_indicators(data_history)
        if data_history is None:
            return "<p>No ADX chart available</p>"
        
        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] ADX of symbol: {self.name_ticker} (Period: {self.data_period})")
        plt.plot(data_history.index, data_history['ADX'], label='ADX')
        plt.legend(loc='upper left')
        plt.axhline(y=25, color='gray', linestyle='--')
        
        plt_buffer = BytesIO()
        plt.savefig(plt_buffer, format='png', bbox_inches='tight')
        plt_buffer.seek(0)
        plt_base64 = b64encode(plt_buffer.read()).decode('utf-8')
        plt_html = f"<img src='data:image/png;base64,{plt_base64}' class='chart-img' onclick='enlargeImage(this)' ondblclick='downsizeImage(this)'>"
        plt.close()
        return plt_html

class VWAP(Stock):
    def __init__(self, name_ticker, data_period='1y'):
        super().__init__(name_ticker)
        self.data_period = data_period
        self.chart_figsize = (20, 12)

    def calculate_vwap(self):
        ticker_data = self.get_history_data()
        if ticker_data is None:
            logging.error(f"No data for VWAP calculation for {self.name_ticker}")
            return None
        
        try:
            tp = (ticker_data['High'] + ticker_data['Low'] + ticker_data['Close']) / 3
            ticker_data['TP'] = tp
            ticker_data['TradedValue'] = ticker_data['TP'] * ticker_data['Volume']
            ticker_data['CumulativeTradedValue'] = ticker_data['TradedValue'].cumsum()
            ticker_data['CumulativeVolume'] = ticker_data['Volume'].cumsum()
            ticker_data['VWAP'] = ticker_data['CumulativeTradedValue'] / ticker_data['CumulativeVolume']
            return ticker_data['VWAP']
        except Exception as e:
            logging.error(f"Error calculating VWAP for {self.name_ticker}: {e}")
            return None

    def plot_chart_vwap_html(self):
        vwap_data = self.calculate_vwap()
        if vwap_data is None:
            logging.error(f"No data for VWAP plot for {self.name_ticker}")
            return "<p>No VWAP chart available</p>"
        
        ticker_data = self.get_history_data()
        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] VWAP of symbol: {self.name_ticker} (Period: {self.data_period})")
        plt.plot(ticker_data.index, ticker_data['Close'], label='Closing Price')
        plt.plot(vwap_data.index, vwap_data, label='VWAP')
        plt.legend(loc='upper left')
        
        plt_buffer = BytesIO()
        plt.savefig(plt_buffer, format='png', bbox_inches='tight')
        plt_buffer.seek(0)
        plt_base64 = b64encode(plt_buffer.read()).decode('utf-8')
        plt_html = f"<img src='data:image/png;base64,{plt_base64}' class='chart-img' onclick='enlargeImage(this)' ondblclick='downsizeImage(this)'>"
        plt.close()
        return plt_html

class StochasticOscillator(Stock):
    def __init__(self, name_ticker, data_period='1y', timeframe=14):
        super().__init__(name_ticker)
        self.data_period = data_period
        self.timeframe = timeframe
        self.chart_figsize = (20, 12)

    def set_timeframe(self, timeframe):
        self.timeframe = timeframe

    def calculate_stochastic_oscillator(self):
        ticker_data = self.get_history_data()
        if ticker_data is None:
            logging.error(f"No data for Stochastic Oscillator calculation for {self.name_ticker}")
            return None, None
        
        try:
            high = ticker_data['High'].rolling(self.timeframe).max()
            low = ticker_data['Low'].rolling(self.timeframe).min()
            k = 100 * (ticker_data['Close'] - low) / (high - low)
            d = k.rolling(3).mean()
            return k, d
        except Exception as e:
            logging.error(f"Error calculating Stochastic Oscillator for {self.name_ticker}: {e}")
            return None, None

    def plot_chart_stochastic_oscillator_html(self):
        k, d = self.calculate_stochastic_oscillator()
        if k is None or d is None:
            logging.error(f"No data for Stochastic Oscillator plot for {self.name_ticker}")
            return "<p>No Stochastic Oscillator chart available</p>"
        
        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] Stochastic Oscillator of symbol: {self.name_ticker} (Period: {self.data_period}, Timeframe: {self.timeframe} days)")
        plt.plot(k, label='%K(Main)')
        plt.plot(d, label='%D(MA)')
        plt.axhline(y=20, color='gray', linestyle='--')
        plt.axhline(y=80, color='gray', linestyle='--')
        plt.legend(loc='upper left')
        
        plt_buffer = BytesIO()
        plt.savefig(plt_buffer, format='png', bbox_inches='tight')
        plt_buffer.seek(0)
        plt_base64 = b64encode(plt_buffer.read()).decode('utf-8')
        plt_html = f"<img src='data:image/png;base64,{plt_base64}' class='chart-img' onclick='enlargeImage(this)' ondblclick='downsizeImage(this)'>"
        plt.close()
        return plt_html

class RSI(Stock):
    def __init__(self, name_ticker, data_period='1y', timeframe=14):
        super().__init__(name_ticker)
        self.data_period = data_period
        self.timeframe = timeframe
        self.chart_figsize = (20, 12)

    def set_timeframe(self, timeframe):
        self.timeframe = timeframe

    def calculate_rsi(self):
        data_history = self.get_history_data()
        if data_history is None:
            logging.error(f"No data for RSI calculation for {self.name_ticker}")
            return None
        
        try:
            delta = data_history['Close'].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ma_up = up.rolling(window=self.timeframe).mean()
            ma_down = down.rolling(window=self.timeframe).mean()
            rs = ma_up / ma_down
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logging.error(f"Error calculating RSI for {self.name_ticker}: {e}")
            return None

    def plot_chart_rsi_html(self):
        data_history = self.get_history_data()
        if data_history is None:
            logging.error(f"No data for RSI plot for {self.name_ticker}")
            return "<p>No RSI chart available</p>"
        
        rsi = self.calculate_rsi()
        if rsi is None:
            return "<p>No RSI chart available</p>"
        
        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] RSI Chart of symbol: {self.name_ticker} (Period: {self.data_period}, Timeframe: {self.timeframe} days)")
        plt.plot(data_history.index, data_history['Close'], label='Closing Price')
        plt.plot(rsi.index, rsi, label='RSI')
        plt.axhline(y=30, color='gray', linestyle='--')
        plt.axhline(y=70, color='gray', linestyle='--')
        plt.legend(loc='upper left')
        
        plt_buffer = BytesIO()
        plt.savefig(plt_buffer, format='png', bbox_inches='tight')
        plt_buffer.seek(0)
        plt_base64 = b64encode(plt_buffer.read()).decode('utf-8')
        plt_html = f"<img src='data:image/png;base64,{plt_base64}' class='chart-img' onclick='enlargeImage(this)' ondblclick='downsizeImage(this)'>"
        plt.close()
        return plt_html

class MADC(Stock):
    def __init__(self, name_ticker, data_period='1y', short_ma=5, long_ma=20):
        super().__init__(name_ticker)
        self.data_period = data_period
        self.chart_figsize = (20, 12)
        self.short_ma = short_ma
        self.long_ma = long_ma

    def calculate_madc(self):
        data_history = self.get_history_data()
        if data_history is None:
            logging.error(f"No data for MADC calculation for {self.name_ticker}")
            return None, None, None
        
        try:
            short_ema = data_history['Close'].ewm(span=self.short_ma, adjust=False).mean()
            long_ema = data_history['Close'].ewm(span=self.long_ma, adjust=False).mean()
            macd_line = short_ema - long_ema
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            ma_diff = macd_line - signal_line
            return macd_line, signal_line, ma_diff
        except Exception as e:
            logging.error(f"Error calculating MADC for {self.name_ticker}: {e}")
            return None, None, None

    def plot_chart_madc_html(self):
        macd_line, signal_line, ma_diff = self.calculate_madc()
        if macd_line is None:
            logging.error(f"No data for MADC plot for {self.name_ticker}")
            return "<p>No MADC chart available</p>"
        
        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] MADC Chart of symbol: {self.name_ticker} (Period: {self.data_period})")
        plt.plot(madc_line, label='MACD Line')
        plt.plot(signal_line, label='Signal Line')
        plt.bar(ma_diff.index, ma_diff, width=0.5, align='center', label=f"MA{self.short_ma} - MA{self.long_ma}", color='gray')
        plt.legend(loc='upper left')
        
        plt_buffer = BytesIO()
        plt.savefig(plt_buffer, format='png', bbox_inches='tight')
        plt_buffer.seek(0)
        plt_base64 = b64encode(plt_buffer.read()).decode('utf-8')
        plt_html = f"<img src='data:image/png;base64,{plt_base64}' class='chart-img' onclick='enlargeImage(this)' ondblclick='downsizeImage(this)'>"
        plt.close()
        return plt_html

class FibonacciRetracement(Stock):
    def __init__(self, name_ticker, data_period='1y'):
        super().__init__(name_ticker)
        self.data_period = data_period
        self.chart_figsize = (20, 12)

    def calculate_swing(self, ticker_data):
        try:
            highest_swing = -1
            lowest_swing = -1
            for i in range(1, ticker_data.shape[0]-1):
                if (ticker_data['High'].iloc[i] > ticker_data['High'].iloc[i-1] and
                    ticker_data['High'].iloc[i] > ticker_data['High'].iloc[i+1] and
                    (highest_swing == -1 or ticker_data['High'].iloc[i] > ticker_data['High'].iloc[highest_swing])):
                    highest_swing = i
                if (ticker_data['Low'].iloc[i] < ticker_data['Low'].iloc[i-1] and
                    ticker_data['Low'].iloc[i] < ticker_data['Low'].iloc[i+1] and
                    (lowest_swing == -1 or ticker_data['Low'].iloc[i] < ticker_data['Low'].iloc[lowest_swing])):
                    lowest_swing = i
            return highest_swing, lowest_swing
        except Exception as e:
            logging.error(f"Error calculating swing for {self.name_ticker}: {e}")
            return -1, -1

    def calculate_fibonacci_levels(self, ticker_data, highest_swing, lowest_swing):
        try:
            ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
            colors = ["black", "red", "green", "blue", "cyan", "magenta", "yellow"]
            levels = []
            max_level = ticker_data['High'].iloc[highest_swing]
            min_level = ticker_data['Low'].iloc[lowest_swing]
            for ratio in ratios:
                if highest_swing > lowest_swing:  # Uptrend
                    levels.append(max_level - (max_level - min_level) * ratio)
                else:  # Downtrend
                    levels.append(min_level + (max_level - min_level) * ratio)
            return levels, colors, ratios
        except Exception as e:
            logging.error(f"Error calculating Fibonacci levels for {self.name_ticker}: {e}")
            return [], [], []

    def plot_chart_fibonacci_retracement_html(self):
        ticker_data = self.get_history_data()
        if ticker_data is None:
            logging.error(f"No data for Fibonacci Retracement plot for {self.name_ticker}")
            return "<p>No Fibonacci Retracement chart available</p>"
        
        highest_swing, lowest_swing = self.calculate_swing(ticker_data)
        if highest_swing == -1 or lowest_swing == -1:
            return "<p>No Fibonacci Retracement chart available</p>"
        
        levels, colors, ratios = self.calculate_fibonacci_levels(ticker_data, highest_swing, lowest_swing)
        if not levels:
            return "<p>No Fibonacci Retracement chart available</p>"
        
        plt.rcParams['figure.figsize'] = self.chart_figsize
        plt.rc('font', size=14)
        plt.plot(ticker_data['Close'])
        for i in range(len(levels)):
            plt.hlines(levels[i], xmin=ticker_data.index[0], xmax=ticker_data.index[-1], 
                      label="{:.1f}%".format(ratios[i] * 100), colors=colors[i], linestyles="dashed")
        plt.legend()
        plt.title(f"[DA] {self.name_ticker.upper()} Stock Data ({self.data_period}) with Fibonacci Retracement Levels")
        
        plt_buffer = BytesIO()
        plt.savefig(plt_buffer, format='png', bbox_inches='tight')
        plt_buffer.seek(0)
        plt_base64 = b64encode(plt_buffer.read()).decode('utf-8')
        plt_html = f"<img src='data:image/png;base64,{plt_base64}' class='chart-img' onclick='enlargeImage(this)' ondblclick='downsizeImage(this)'>"
        plt.close()
        return plt_html

class OBV(Stock):
    def __init__(self, name_ticker, data_period='1y'):
        super().__init__(name_ticker)
        self.data_period = data_period
        self.chart_figsize = (20, 12)

    def calculate_obv(self):
        ticker_data = self.get_history_data()
        if ticker_data is None:
            logging.error(f"No data for OBV calculation for {self.name_ticker}")
            return None
        
        try:
            obv = []
            prev_obv = 0
            for i in range(1, len(ticker_data)):
                if ticker_data['Close'].iloc[i] > ticker_data['Close'].iloc[i-1]:
                    current_obv = prev_obv + ticker_data['Volume'].iloc[i]
                elif ticker_data['Close'].iloc[i] < ticker_data['Close'].iloc[i-1]:
                    current_obv = prev_obv - ticker_data['Volume'].iloc[i]
                else:
                    current_obv = prev_obv
                obv.append(current_obv)
                prev_obv = current_obv
            return pd.Series(obv, index=ticker_data.index[1:])
        except Exception as e:
            logging.error(f"Error calculating OBV for {self.name_ticker}: {e}")
            return None

    def plot_chart_obv_html(self):
        obv_data = self.calculate_obv()
        if obv_data is None:
            logging.error(f"No data for OBV plot for {self.name_ticker}")
            return "<p>No OBV chart available</p>"
        
        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] OBV of symbol: {self.name_ticker} (Period: {self.data_period})")
        plt.plot(obv_data.index, obv_data, label='OBV')
        plt.axhline(y=0, color='black', linestyle='--')
        plt.legend(loc='upper left')
        
        plt_buffer = BytesIO()
        plt.savefig(plt_buffer, format='png', bbox_inches='tight')
        plt_buffer.seek(0)
        plt_base64 = b64encode(plt_buffer.read()).decode('utf-8')
        plt_html = f"<img src='data:image/png;base64,{plt_base64}' class='chart-img' onclick='enlargeImage(this)' ondblclick='downsizeImage(this)'>"
        plt.close()
        return plt_html

class AccumulationDistributionLine(Stock):
    def __init__(self, name_ticker, data_period='1y'):
        super().__init__(name_ticker)
        self.data_period = data_period
        self.chart_figsize = (20, 12)

    def calculate_adl(self):
        self.df = self.get_history_data()
        if self.df is None:
            logging.error(f"No data for ADL calculation for {self.name_ticker}")
            return None
        
        try:
            self.df['CMF Multiplier'] = ((self.df['Close'] - self.df['Low']) - (self.df['High'] - self.df['Close'])) / (self.df['High'] - self.df['Low'])
            self.df['CMF Volume'] = self.df['CMF Multiplier'] * self.df['Volume']
            self.df['ADL'] = self.df['CMF Volume'].cumsum()
            return self.df[['ADL']]
        except Exception as e:
            logging.error(f"Error calculating ADL for {self.name_ticker}: {e}")
            return None

    def plot_chart_ADL_html(self):
        adl_data = self.calculate_adl()
        if adl_data is None:
            logging.error(f"No data for ADL plot for {self.name_ticker}")
            return "<p>No ADL chart available</p>"
        
        plt.figure(figsize=self.chart_figsize)
        plt.title(f"[DA] ADL of symbol: {self.name_ticker} (Period: {self.data_period})")
        plt.plot(adl_data.index, adl_data['ADL'], label='ADL')
        plt.legend(loc='upper left')
        
        plt_buffer = BytesIO()
        plt.savefig(plt_buffer, format='png', bbox_inches='tight')
        plt_buffer.seek(0)
        plt_base64 = b64encode(plt_buffer.read()).decode('utf-8')
        plt_html = f"<img src='data:image/png;base64,{plt_base64}' class='chart-img' onclick='enlargeImage(this)' ondblclick='downsizeImage(this)'>"
        plt.close()
        return plt_html

class Analysis_TA(BollingerBands, MovingAverage, ADX, VWAP, StochasticOscillator, RSI, MADC, FibonacciRetracement, OBV, AccumulationDistributionLine):
    def __init__(self, name_ticker, data_period='1y'):
        super().__init__(name_ticker)
        self.data_period = data_period

class SharpeRatio(Stock):
    def __init__(self, name_ticker, data_period='1y', risk_free_rate=0.05):
        super().__init__(name_ticker)
        self.data_period = data_period
        self.risk_free_rate = risk_free_rate
        self.stock_data = self.get_history_data()
        if self.stock_data is not None:
            self.stock_data = self.stock_data['Close']
            self.daily_returns = self.stock_data.pct_change().dropna()
            self.annual_returns = self.daily_returns.mean() * 252
            self.annual_volatility = self.daily_returns.std() * np.sqrt(252)
            self.sharpe_ratio = (self.annual_returns - self.risk_free_rate) / self.annual_volatility
        else:
            self.daily_returns = None
            self.annual_returns = None
            self.annual_volatility = None
            self.sharpe_ratio = None
        self.chart_figsize = (20, 12)

    def plot_returns_html(self):
        if self.stock_data is None:
            logging.error(f"No data for returns plot for {self.name_ticker}")
            return "<p>No returns chart available</p>"
        
        cumulative_returns = (self.stock_data / self.stock_data.iloc[0] - 1) * 100
        plt.figure(figsize=self.chart_figsize)
        plt.plot(cumulative_returns)
        plt.title(f"Cumulative Returns of {self.name_ticker.upper()} ({self.data_period})")
        plt.ylabel("Cumulative Returns (%)")
        
        plt_buffer = BytesIO()
        plt.savefig(plt_buffer, format='png', bbox_inches='tight')
        plt_buffer.seek(0)
        plt_base64 = b64encode(plt_buffer.read()).decode('utf-8')
        plt_html = f"<img src='data:image/png;base64,{plt_base64}' class='chart-img' onclick='enlargeImage(this)' ondblclick='downsizeImage(this)'>"
        plt.close()
        return plt_html

    def get_metrics(self):
        if self.stock_data is None:
            logging.error(f"No data for metrics calculation for {self.name_ticker}")
            return "No metrics available"
        
        try:
            total_return = self.stock_data.iloc[-1] / self.stock_data.iloc[0] - 1
            cumulative_return = (self.stock_data.iloc[-1] / self.stock_data.iloc[0] - 1) * 100
            max_drawdown = (self.stock_data / self.stock_data.cummax() - 1).min() * 100
            monthly_returns = self.stock_data.resample('M').ffill().pct_change().dropna()
            sharpe_ratio = (monthly_returns.mean() - self.risk_free_rate / 12) / monthly_returns.std() * np.sqrt(12)
            return (f"{self.name_ticker.upper()} ({self.data_period}) Monthly Sharpe Ratio: {sharpe_ratio:.2f}\n"
                    f"Total Return: {total_return:.2f}\n"
                    f"Cumulative Return: {cumulative_return:.2f}%\n"
                    f"Max Drawdown: {max_drawdown:.2f}%")
        except Exception as e:
            logging.error(f"Error calculating metrics for {self.name_ticker}: {e}")
            return "No metrics available"

class RRnRRR(Stock):
    def __init__(self, name_ticker):
        super().__init__(name_ticker)
        self.data = self.get_history_data()

    def get_rr(self, entry_price, stop_loss):
        if self.data is None:
            logging.error(f"No data for RR calculation for {self.name_ticker}")
            return 0
        
        try:
            reward = abs(self.data['Close'].iloc[-1] - entry_price)
            risk = abs(entry_price - stop_loss)
            if reward == 0 or risk == 0:
                return 0
            return reward / risk
        except Exception as e:
            logging.error(f"Error calculating RR for {self.name_ticker}: {e}")
            return 0

    def get_rrr(self, entry_price, stop_loss, target_price):
        if self.data is None:
            logging.error(f"No data for RRR calculation for {self.name_ticker}")
            return 0
        
        try:
            reward = abs(target_price - entry_price)
            risk = abs(entry_price - stop_loss)
            if reward == 0 or risk == 0:
                return 0
            return reward / risk
        except Exception as e:
            logging.error(f"Error calculating RRR for {self.name_ticker}: {e}")
            return 0

class BullBearIndicator(Stock):
    def __init__(self, name_ticker, data_period='1y'):
        super().__init__(name_ticker)
        self.data_period = data_period
        self.stock_data = self.get_history_data()
        self.short_window = 20
        self.long_window = 50
        self.tf_rsi = 14
        self.std_dev = 2
        self.tf_bb = 20
        if self.stock_data is None or self.stock_data.empty:
            self.stock_data = None
        else:
            self.stock_data = self.stock_data.dropna()

    def calculate_moving_averages(self):
        if self.stock_data is None:
            return
        try:
            self.stock_data['SMA_short'] = self.stock_data['Close'].rolling(window=self.short_window).mean()
            self.stock_data['SMA_long'] = self.stock_data['Close'].rolling(window=self.long_window).mean()
        except Exception as e:
            logging.error(f"Error calculating moving averages for {self.name_ticker}: {e}")

    def is_bullish_ma(self):
        if self.stock_data is None:
            return False
        try:
            self.calculate_moving_averages()
            last_price = self.stock_data['Close'].iloc[-1]
            sma_short = self.stock_data['SMA_short'].iloc[-1]
            sma_long = self.stock_data['SMA_long'].iloc[-1]
            if sma_short > sma_long and last_price > sma_short:
                return True
            return False
        except Exception as e:
            logging.error(f"Error in is_bullish_ma for {self.name_ticker}: {e}")
            return False

    def is_bearish_ma(self):
        if self.stock_data is None:
            return False
        try:
            self.calculate_moving_averages()
            last_price = self.stock_data['Close'].iloc[-1]
            sma_short = self.stock_data['SMA_short'].iloc[-1]
            sma_long = self.stock_data['SMA_long'].iloc[-1]
            if sma_short < sma_long and last_price < sma_short:
                return True
            return False
        except Exception as e:
            logging.error(f"Error in is_bearish_ma for {self.name_ticker}: {e}")
            return False

    def calculate_sma(self, period):
        if self.stock_data is None:
            return None
        try:
            return self.stock_data['Close'].rolling(window=period).mean()
        except Exception as e:
            logging.error(f"Error calculating SMA for {self.name_ticker}: {e}")
            return None

    def is_bullish_bd(self):
        if self.stock_data is None:
            return False
        try:
            sma_short = self.calculate_sma(self.short_window)
            sma_long = self.calculate_sma(self.long_window)
            if sma_short is None or sma_long is None:
                return False
            if sma_short.iloc[-1] > sma_long.iloc[-1] and sma_short.iloc[-2] < sma_long.iloc[-2]:
                return True
            return False
        except Exception as e:
            logging.error(f"Error in is_bullish_bd for {self.name_ticker}: {e}")
            return False

    def is_bearish_bd(self):
        if self.stock_data is None:
            return False
        try:
            sma_short = self.calculate_sma(self.short_window)
            sma_long = self.calculate_sma(self.long_window)
            if sma_short is None or sma_long is None:
                return False
            if sma_short.iloc[-1] < sma_long.iloc[-1] and sma_short.iloc[-2] > sma_long.iloc[-2]:
                return True
            return False
        except Exception as e:
            logging.error(f"Error in is_bearish_bd for {self.name_ticker}: {e}")
            return False

    def calculate_rsi(self, timeframe):
        if self.stock_data is None:
            return None
        try:
            delta = self.stock_data['Close'].diff().dropna()
            gains = delta.copy()
            losses = delta.copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            avg_gain = gains.rolling(timeframe).mean().dropna()
            avg_loss = -losses.rolling(timeframe).mean().dropna()
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
            return rsi
        except Exception as e:
            logging.error(f"Error calculating RSI for {self.name_ticker}: {e}")
            return None

    def is_bullish_rsi(self):
        if self.stock_data is None:
            return False
        try:
            rsi = self.calculate_rsi(self.tf_rsi)
            if rsi is None or rsi.empty:
                return False
            if rsi.iloc[-1] < 30 and rsi.iloc[-2] > 30:
                return True
            return False
        except Exception as e:
            logging.error(f"Error in is_bullish_rsi for {self.name_ticker}: {e}")
            return False

    def is_bearish_rsi(self):
        if self.stock_data is None:
            return False
        try:
            rsi = self.calculate_rsi(self.tf_rsi)
            if rsi is None or rsi.empty:
                return False
            if rsi.iloc[-1] > 70 and rsi.iloc[-2] < 70:
                return True
            return False
        except Exception as e:
            logging.error(f"Error in is_bearish_rsi for {self.name_ticker}: {e}")
            return False

    def calculate_bollinger_bands(self):
        if self.stock_data is None:
            return
        try:
            self.stock_data['MA'] = self.stock_data['Close'].rolling(window=self.tf_bb).mean()
            self.stock_data['STD'] = self.stock_data['Close'].rolling(window=self.tf_bb).std()
            self.stock_data['Upper Band'] = self.stock_data['MA'] + (self.stock_data['STD'] * self.std_dev)
            self.stock_data['Lower Band'] = self.stock_data['MA'] - (self.stock_data['STD'] * self.std_dev)
        except Exception as e:
            logging.error(f"Error calculating Bollinger Bands for {self.name_ticker}: {e}")

    def is_bullish_bollinger_bands(self):
        if self.stock_data is None:
            return False
        try:
            self.calculate_bollinger_bands()
            if self.stock_data['Close'].iloc[-1] < self.stock_data['Lower Band'].iloc[-1]:
                return True
            return False
        except Exception as e:
            logging.error(f"Error in is_bullish_bollinger_bands for {self.name_ticker}: {e}")
            return False

    def is_bearish_bollinger_bands(self):
        if self.stock_data is None:
            return False
        try:
            self.calculate_bollinger_bands()
            if self.stock_data['Close'].iloc[-1] > self.stock_data['Upper Band'].iloc[-1]:
                return True
            return False
        except Exception as e:
            logging.error(f"Error in is_bearish_bollinger_bands for {self.name_ticker}: {e}")
            return False

def check_bullish(name_ticker, no_signal=2, data_period='6mo'):
    try:
        bb_signal = BullBearIndicator(name_ticker, data_period)
        if bb_signal.stock_data is None:
            logging.warning(f"No data available for {name_ticker}. Skipping...")
            return None
        indicators = [
            {'name': 'Moving Average', 'is_bullish': bb_signal.is_bullish_ma()},
            {'name': 'Breakdown', 'is_bullish': bb_signal.is_bullish_bd()},
            {'name': 'OversoldSignal(RSI)', 'is_bullish': bb_signal.is_bullish_rsi()},
            {'name': 'Bollinger Bands', 'is_bullish': bb_signal.is_bullish_bollinger_bands()}
        ]
        tickers = [name_ticker for i in indicators if i['is_bullish']]
        return name_ticker if len(tickers) >= no_signal else None
    except Exception as e:
        logging.error(f"Error in check_bullish for {name_ticker}: {e}")
        return None

def check_bearish(name_ticker, no_signal=2, data_period='6mo'):
    try:
        bb_signal = BullBearIndicator(name_ticker, data_period)
        if bb_signal.stock_data is None:
            logging.warning(f"No data available for {name_ticker}. Skipping...")
            return None
        indicators = [
            {'name': 'Moving Average', 'is_bearish': bb_signal.is_bearish_ma()},
            {'name': 'Breakdown', 'is_bearish': bb_signal.is_bearish_bd()},
            {'name': 'OversoldSignal(RSI)', 'is_bearish': bb_signal.is_bearish_rsi()},
            {'name': 'Bollinger Bands', 'is_bearish': bb_signal.is_bearish_bollinger_bands()}
        ]
        tickers = [name_ticker for i in indicators if i['is_bearish']]
        return name_ticker if len(tickers) >= no_signal else None
    except Exception as e:
        logging.error(f"Error in check_bearish for {name_ticker}: {e}")
        return None

def scan_sp500_bb():
    lst_bullish = []
    lst_bearish = []
    html_content = """
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8'>
        <meta name='viewport' content='width=device-width, initial-scale=1.0'>
        <title>S&P 500 Stock Analysis Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .chart-img { max-width: 100%; height: auto; cursor: pointer; }
            .chart-img.enlarged { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); max-width: 90%; max-height: 90%; z-index: 1000; }
            .section { margin-bottom: 40px; }
        </style>
        <script>
            function enlargeImage(img) {
                img.classList.toggle('enlarged');
            }
            function downsizeImage(img) {
                img.classList.remove('enlarged');
            }
        </script>
    </head>
    <body>
        <h1>S&P 500 Stock Analysis Report</h1>
        <div class='section'>
            <h2>Bullish Stocks</h2>
            <table>
                <tr><th>Ticker</th><th>Prediction Date</th><th>Model</th><th>Predicted Close</th><th>Actual Close</th></tr>
    """
    
    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def fetch_sp500_tickers():
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            dfs = pd.read_html(url, header=0)
            sp500_df = dfs[0]
            sp500_tickers = sp500_df['Symbol'].tolist()
            sp500_tickers = [ticker.replace(".", "-") for ticker in sp500_tickers]
            return sp500_tickers
        except Exception as e:
            logging.error(f"Error fetching S&P 500 tickers: {e}")
            return []

    sp500_tickers = fetch_sp500_tickers()
    if not sp500_tickers:
        logging.error("No S&P 500 tickers retrieved. Generating empty report.")
        html_content += "</table></div><div class='section'><h2>Bearish Stocks</h2><table><tr><th>Ticker</th><th>Prediction Date</th><th>Model</th><th>Predicted Close</th><th>Actual Close</th></tr></table></div></body></html>"
        with open('index.html', 'w') as f:
            f.write(html_content)
        return [], []

    # Limit to a subset of tickers to avoid rate limits and long runtime
    sample_tickers = sp500_tickers[:10]  # Adjust as needed
    for ticker in sample_tickers:
        try:
            if check_bullish(ticker):
                lst_bullish.append(ticker)
                analysis = Analysis_TA(ticker)
                sharpe = SharpeRatio(ticker)
                html_content += f"<tr><td colspan='5'><h3>{ticker}</h3>{analysis.get_stock_info_html()}</td></tr>"
                html_content += analysis.ml_RFR_html()
                html_content += f"<tr><td colspan='5'>{sharpe.get_metrics()}</td></tr>"
                html_content += f"<tr><td colspan='5'>{analysis.plot_chart_MovingAverage_html()}</td></tr>"
                html_content += f"<tr><td colspan='5'>{analysis.plot_chart_BollingerBands_html()}</td></tr>"
                html_content += f"<tr><td colspan='5'>{analysis.plot_chart_ADX_html()}</td></tr>"
                html_content += f"<tr><td colspan='5'>{analysis.plot_chart_vwap_html()}</td></tr>"
                html_content += f"<tr><td colspan='5'>{analysis.plot_chart_stochastic_oscillator_html()}</td></tr>"
                html_content += f"<tr><td colspan='5'>{analysis.plot_chart_rsi_html()}</td></tr>"
                html_content += f"<tr><td colspan='5'>{analysis.plot_chart_madc_html()}</td></tr>"
                html_content += f"<tr><td colspan='5'>{analysis.plot_chart_fibonacci_retracement_html()}</td></tr>"
                html_content += f"<tr><td colspan='5'>{analysis.plot_chart_obv_html()}</td></tr>"
                html_content += f"<tr><td colspan='5'>{analysis.plot_chart_ADL_html()}</td></tr>"
                html_content += f"<tr><td colspan='5'>{sharpe.plot_returns_html()}</td></tr>"
            if check_bearish(ticker):
                lst_bearish.append(ticker)
        except Exception as e:
            logging.error(f"Error processing ticker {ticker}: {e}")
            continue

    html_content += "</table></div><div class='section'><h2>Bearish Stocks</h2><table><tr><th>Ticker</th><th>Prediction Date</th><th>Model</th><th>Predicted Close</th><th>Actual Close</th></tr>"
    for ticker in lst_bearish:
        try:
            analysis = Analysis_TA(ticker)
            sharpe = SharpeRatio(ticker)
            html_content += f"<tr><td colspan='5'><h3>{ticker}</h3>{analysis.get_stock_info_html()}</td></tr>"
            html_content += analysis.ml_RFR_html()
            html_content += f"<tr><td colspan='5'>{sharpe.get_metrics()}</td></tr>"
            html_content += f"<tr><td colspan='5'>{analysis.plot_chart_MovingAverage_html()}</td></tr>"
            html_content += f"<tr><td colspan='5'>{analysis.plot_chart_BollingerBands_html()}</td></tr>"
            html_content += f"<tr><td colspan='5'>{analysis.plot_chart_ADX_html()}</td></tr>"
            html_content += f"<tr><td colspan='5'>{analysis.plot_chart_vwap_html()}</td></tr>"
            html_content += f"<tr><td colspan='5'>{analysis.plot_chart_stochastic_oscillator_html()}</td></tr>"
            html_content += f"<tr><td colspan='5'>{analysis.plot_chart_rsi_html()}</td></tr>"
            html_content += f"<tr><td colspan='5'>{analysis.plot_chart_madc_html()}</td></tr>"
            html_content += f"<tr><td colspan='5'>{analysis.plot_chart_fibonacci_retracement_html()}</td></tr>"
            html_content += f"<tr><td colspan='5'>{analysis.plot_chart_obv_html()}</td></tr>"
            html_content += f"<tr><td colspan='5'>{analysis.plot_chart_ADL_html()}</td></tr>"
            html_content += f"<tr><td colspan='5'>{sharpe.plot_returns_html()}</td></tr>"
        except Exception as e:
            logging.error(f"Error processing bearish ticker {ticker}: {e}")
            continue

    html_content += "</table></div></body></html>"
    try:
        with open('index.html', 'w') as f:
            f.write(html_content)
    except Exception as e:
        logging.error(f"Error writing index.html: {e}")
    
    return lst_bullish, lst_bearish

if __name__ == "__main__":
    scan_sp500_bb()
