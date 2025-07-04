from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from io import BytesIO
from base64 import b64encode
from IPython.core.display import HTML
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Stock:
    symbol_type = "stock"
    default_data_period = '1y'

    def __init__(self, name_ticker):
        self.name_ticker = name_ticker
        self.data_period = self.default_data_period
        self._history_data = None

    def set_data_period(self, data_period):
        if data_period in ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']:
            self.data_period = data_period
        else:
            logging.error(f"Invalid data period: {data_period}")
            raise ValueError("Invalid data period")

    def get_current_price(self):
        try:
            ticker_data = yf.Ticker(self.name_ticker)
            info = ticker_data.info
            for key in ["currentPrice", "ask", "navPrice"]:  # Adjusted key order
                if key in info and info[key] is not None:
                    return info[key]
            logging.warning(f"No valid price found for {self.name_ticker}")
            return None
        except Exception as e:
            logging.error(f"Could not retrieve current price for {self.name_ticker}: {e}")
            return None

    def get_price_target_date(self, target_date):
        try:
            date_obj = datetime.strptime(target_date, '%Y-%m-%d')
            ticker_data = yf.download(self.name_ticker, start=target_date, end=date_obj + timedelta(days=1), progress=False, auto_adjust=True)
            if ticker_data.empty or 'Close' not in ticker_data.columns:
                logging.error(f"No data for {self.name_ticker} on {target_date}")
                return None
            return (target_date, ticker_data['Close'].iloc[0])
        except Exception as e:
            logging.error(f"Could not retrieve price for {self.name_ticker} on {target_date}: {e}")
            return None

    def get_history_data(self):
        if self._history_data is not None:
            return self._history_data
        try:
            data_history = yf.download(self.name_ticker, period=self.data_period, progress=False, auto_adjust=True)
            if data_history.empty or 'Close' not in data_history.columns:
                logging.error(f"No valid data found for {self.name_ticker}")
                return None
            self._history_data = data_history
            return data_history
        except Exception as e:
            logging.error(f"Error retrieving data for {self.name_ticker}: {e}")
            return None

    def get_stock_info_html(self):
        price = self.get_current_price()
        price_str = f"{price:.2f}" if price is not None else "N/A"
        return f"""
            <p>Symbol Type: {self.symbol_type}</p>
            <p>Ticker: {self.name_ticker}</p>
            <p>Current Data Period: {self.data_period}</p>
            <p>Latest Price: {price_str}</p>
        """

    def ml_RFR_html(self):
        try:
            stock_data = self.get_history_data()
            if stock_data is None or stock_data.empty:
                logging.error(f"No data for RandomForestRegressor for {self.name_ticker}")
                return "<tr><td colspan='8'>No data available for ML analysis</td></tr>"

            feature_cols = ["Open", "High", "Low", "Volume"]
            target_col = "Close"

            # Shift target to next day's close
            stock_data['Next_Close'] = stock_data['Close'].shift(-1)
            stock_data = stock_data.dropna(subset=['Next_Close'])

            if stock_data.empty:
                return "<tr><td colspan='8'>Insufficient data for ML analysis</td></tr>"

            x = stock_data[feature_cols]
            y = stock_data['Next_Close']

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=55)
            rf_model = RandomForestRegressor(n_estimators=400, random_state=55)
            rf_model.fit(x_train, y_train)

            y_pred = rf_model.predict(x_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)

            last_day_features = stock_data[feature_cols].iloc[-1].values.reshape(1, -1)
            predicted_next_close = rf_model.predict(last_day_features)[0]
            next_date = stock_data.index[-1] + timedelta(days=1)

            return f"""
                <tr>
                    <td>{self.name_ticker}</td>
                    <td>{next_date.strftime('%Y-%m-%d')}</td>
                    <td>RandomForestRegressor</td>
                    <td>{round(predicted_next_close, 2)}</td>
                    <td>N/A</td>
                    <td>{round(r2, 4)}</td>
                    <td>{round(mse, 4)}</td>
                    <td>{round(mape, 4)}</td>
                </tr>
            """
        except Exception as e:
            logging.error(f"Error in ML analysis for {self.name_ticker}: {e}")
            return "<tr><td colspan='8'>Error in ML analysis</td></tr>"

    def _plot_to_html(self, plot_func, title):
        plt.figure(figsize=(20, 12))
        plot_func()
        plt.title(title)
        plt.legend(loc='upper left')
        plt_buffer = BytesIO()
        plt.savefig(plt_buffer, format='png')
        plt_buffer.seek(0)
        plt_base64 = b64encode(plt_buffer.read()).decode('utf-8')
        plt.close()
        return f"<img src='data:image/png;base64,{plt_base64}' class='chart-img' onclick='enlargeImage(this)' ondblclick='resetImage(this)'>"

    def __str__(self):
        price = self.get_current_price()
        price_str = f"{price:.2f}" if price is not None else "N/A"
        return f"[***** Symbol Info *****]\nSymbol Type: {self.symbol_type}\nTicker: {self.name_ticker}\nCurrent Data Period: {self.data_period}\nLatest Price: {price_str}"

class MovingAverage(Stock):
    def __init__(self, name_ticker, data_period='1y'):
        super().__init__(name_ticker)
        self.set_data_period(data_period)
        self.chart_figsize = (20, 12)

    def plot_chart_MovingAverage_html(self):
        data_history = self.get_history_data()
        if data_history is None or data_history.empty or 'Close' not in data_history.columns or len(data_history) < 200:
            return "<p>Insufficient data for Moving Average chart</p>"

        data_history['MA5'] = data_history['Close'].rolling(window=5).mean()
        data_history['MA10'] = data_history['Close'].rolling(window=10).mean()
        data_history['MA50'] = data_history['Close'].rolling(window=50).mean()
        data_history['MA200'] = data_history['Close'].rolling(window=200).mean()

        def plot_func():
            plt.plot(data_history.index, data_history['Close'], label='Closing Price')
            plt.plot(data_history.index, data_history['MA5'], label='MA5')
            plt.plot(data_history.index, data_history['MA10'], label='MA10')
            plt.plot(data_history.index, data_history['MA50'], label='MA50')
            plt.plot(data_history.index, data_history['MA200'], label='MA200')

        return self._plot_to_html(plot_func, f"[DA] MA Chart of symbol: {self.name_ticker} (Period: {self.data_period})")

class BollingerBands(Stock):
    def __init__(self, name_ticker, data_period='1y'):
        super().__init__(name_ticker)
        self.set_data_period(data_period)
        self.chart_figsize = (20, 12)

    def plot_chart_BollingerBands_html(self):
        data_history = self.get_history_data()
        if data_history is None or data_history.empty or 'Close' not in data_history.columns or len(data_history) < 20:
            return "<p>Insufficient data for Bollinger Bands chart</p>"

        data_history['MA20'] = data_history['Close'].rolling(window=20).mean()
        data_history['20dSTD'] = data_history['Close'].rolling(window=20).std()
        data_history['UpperBand'] = data_history['MA20'] + (data_history['20dSTD'] * 2)
        data_history['LowerBand'] = data_history['MA20'] - (data_history['20dSTD'] * 2)

        def plot_func():
            plt.plot(data_history.index, data_history['Close'], label='Closing Price')
            plt.plot(data_history.index, data_history['MA20'], label='20 Day Moving Average')
            plt.plot(data_history.index, data_history['UpperBand'], label='Upper Bollinger Band')
            plt.plot(data_history.index, data_history['LowerBand'], label='Lower Bollinger Band')
            plt.fill_between(data_history.index, data_history['UpperBand'], data_history['LowerBand'], alpha=0.1)

        return self._plot_to_html(plot_func, f"[DA] Bollinger Bands of symbol: {self.name_ticker} (Period: {self.data_period})")

class ADX(Stock):
    def __init__(self, name_ticker, data_period='1y'):
        super().__init__(name_ticker)
        self.set_data_period(data_period)
        self.chart_figsize = (20, 12)

    def calculate_true_range(self, df):
        if df is None or df.empty:
            return df
        df['TR1'] = abs(df['High'] - df['Low'])
        df['TR2'] = abs(df['High'] - df['Close'].shift())
        df['TR3'] = abs(df['Low'] - df['Close'].shift())
        df['TrueRange'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
        return df.drop(['TR1', 'TR2', 'TR3'], axis=1)

def calculate_directional_indicators(self, df):
    if df is None or df.empty:
        return pd.DataFrame()
    df['PlusDM'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), 
                            df['High'] - df['High'].shift(1), 0)
    df['MinusDM'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), 
                             df['Low'].shift(1) - df['Low'], 0)
    df['TR'] = df['TrueRange']
    df['SmoothedPlusDM'] = df['PlusDM'].rolling(window=14).sum()
    df['SmoothedMinusDM'] = df['MinusDM'].rolling(window=14).sum()
    df['SmoothedTR'] = df['TR'].rolling(window=14).sum()
    df['PlusDI'] = 100 * df['SmoothedPlusDM'] / (df['SmoothedTR'] + 1e-10)
    df['MinusDI'] = 100 * df['SmoothedMinusDM'] / (df['SmoothedTR'] + 1e-10)
    df['DX'] = 100 * abs(df['PlusDI'] - df['MinusDI']) / (df['PlusDI'] + df['MinusDI'] + 1e-10)
    df['ADX'] = df['DX'].rolling(window=14).mean()
    return df[['PlusDI', 'MinusDI', 'ADX']]

    def plot_chart_ADX_html(self):
        data_history = self.get_history_data()
        if data_history is None or data_history.empty or len(data_history) < 28:
            return "<p>Insufficient data for ADX chart</p>"
        data_history = self.calculate_true_range(data_history)
        data_history = self.calculate_directional_indicators(data_history)

        def plot_func():
            plt.plot(data_history.index, data_history['ADX'], label='ADX')
            plt.axhline(y=25, color='gray', linestyle='--')

        return self._plot_to_html(plot_func, f"[DA] ADX of symbol: {self.name_ticker} (Period: {self.data_period})")

class VWAP(Stock):
    def __init__(self, name_ticker, data_period='1y'):
        super().__init__(name_ticker)
        self.set_data_period(data_period)
        self.chart_figsize = (20, 12)

    def calculate_vwap(self):
        ticker_data = self.get_history_data()
        if ticker_data is None or ticker_data.empty:
            return pd.Series()
        tp = (ticker_data['High'] + ticker_data['Low'] + ticker_data['Close']) / 3
        ticker_data['TP'] = tp
        ticker_data['TradedValue'] = ticker_data['TP'] * ticker_data['Volume']
        ticker_data['CumulativeTradedValue'] = ticker_data['TradedValue'].cumsum()
        ticker_data['CumulativeVolume'] = ticker_data['Volume'].cumsum()
        ticker_data['VWAP'] = ticker_data['CumulativeTradedValue'] / (ticker_data['CumulativeVolume'] + 1e-10)
        return ticker_data['VWAP']

    def plot_chart_vwap_html(self):
        vwap_data = self.calculate_vwap()
        data_history = self.get_history_data()
        if vwap_data.empty or data_history is None:
            return "<p>No data available for VWAP chart</p>"

        def plot_func():
            plt.plot(vwap_data.index, data_history['Close'], label='Closing Price')
            plt.plot(vwap_data.index, vwap_data, label='VWAP')

        return self._plot_to_html(plot_func, f"[DA] VWAP of symbol: {self.name_ticker} (Period: {self.data_period})")

class StochasticOscillator(Stock):
    def __init__(self, name_ticker, data_period='1y', timeframe=14):
        super().__init__(name_ticker)
        self.set_data_period(data_period)
        self.timeframe = timeframe
        self.chart_figsize = (20, 12)

    def set_timeframe(self, timeframe):
        if isinstance(timeframe, int) and timeframe > 0:
            self.timeframe = timeframe
        else:
            logging.error(f"Invalid timeframe: {timeframe}")
            raise ValueError("Timeframe must be a positive integer")

    def calculate_stochastic_oscillator(self):
        ticker_data = self.get_history_data()
        if ticker_data is None or ticker_data.empty or len(ticker_data) < self.timeframe:
            return pd.Series(), pd.Series()
        high = ticker_data['High'].rolling(self.timeframe).max()
        low = ticker_data['Low'].rolling(self.timeframe).min()
        k = 100 * (ticker_data['Close'] - low) / (high - low + 1e-10)
        d = k.rolling(3).mean()
        return k, d

    def plot_chart_stochastic_oscillator_html(self):
        k, d = self.calculate_stochastic_oscillator()
        if k.empty or d.empty:
            return "<p>Insufficient data for Stochastic Oscillator chart</p>"

        def plot_func():
            plt.plot(k, label='%K(Main)')
            plt.plot(d, label='%D(MA)')
            plt.axhline(y=20, color='gray', linestyle='--')
            plt.axhline(y=80, color='gray', linestyle='--')

        return self._plot_to_html(plot_func, f"[DA] Stochastic Oscillator of symbol: {self.name_ticker} (Period: {self.data_period}, Timeframe: {self.timeframe} days)")

class RSI(Stock):
    def __init__(self, name_ticker, data_period='1y', timeframe=14):
        super().__init__(name_ticker)
        self.set_data_period(data_period)
        self.timeframe = timeframe
        self.chart_figsize = (20, 12)

    def set_timeframe(self, timeframe):
        if isinstance(timeframe, int) and timeframe > 0:
            self.timeframe = timeframe
        else:
            logging.error(f"Invalid timeframe: {timeframe}")
            raise ValueError("Timeframe must be a positive integer")

    def calculate_rsi(self):
        data_history = self.get_history_data()
        if data_history is None or data_history.empty or 'Close' not in data_history.columns or len(data_history) < self.timeframe:
            return pd.Series()
        delta = data_history['Close'].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.rolling(window=self.timeframe).mean()
        ma_down = down.rolling(window=self.timeframe).mean()
        rs = ma_up / (ma_down + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def plot_chart_rsi_html(self):
        data_history = self.get_history_data()
        if data_history is None or data_history.empty or 'Close' not in data_history.columns or len(data_history) < self.timeframe:
            return "<p>Insufficient data for RSI chart</p>"
        rsi = self.calculate_rsi()

        def plot_func():
            plt.plot(data_history.index, data_history['Close'], label='Closing Price')
            plt.plot(rsi.index, rsi, label='RSI')
            plt.axhline(y=30, color='gray', linestyle='--')
            plt.axhline(y=70, color='gray', linestyle='--')

        return self._plot_to_html(plot_func, f"[DA] RSI Chart of symbol: {self.name_ticker} (Period: {self.data_period}, Timeframe: {self.timeframe} days)")

class MADC(Stock):
    def __init__(self, name_ticker, data_period='1y', short_ma=5, long_ma=20):
        super().__init__(name_ticker)
        self.set_data_period(data_period)
        self.chart_figsize = (20, 12)
        self.short_ma = short_ma
        self.long_ma = long_ma

    def calculate_madc(self):
        data_history = self.get_history_data()
        if data_history is None or data_history.empty or len(data_history) < self.long_ma:
            return pd.Series(), pd.Series(), pd.Series()
        short_ema = data_history['Close'].ewm(span=self.short_ma, adjust=False).mean()
        long_ema = data_history['Close'].ewm(span=self.long_ma, adjust=False).mean()
        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        ma_diff = macd_line - signal_line
        return macd_line, signal_line, ma_diff

    def plot_chart_madc_html(self):
        macd_line, signal_line, ma_diff = self.calculate_madc()
        if macd_line.empty:
            return "<p>Insufficient data for MADC chart</p>"

        def plot_func():
            plt.plot(macd_line, label='MACD Line')
            plt.plot(signal_line, label='Signal Line')
            plt.bar(ma_diff.index, ma_diff, width=0.5, align='center', label=f"MA{self.short_ma} - MA{self.long_ma}", color='gray')

        return self._plot_to_html(plot_func, f"[DA] MADC Chart of symbol: {self.name_ticker} (Period: {self.data_period})")

class FibonacciRetracement(Stock):
    def __init__(self, name_ticker, data_period='1y'):
        super().__init__(name_ticker)
        self.set_data_period(data_period)
        self.chart_figsize = (20, 12)

    def calculate_swing(self, ticker_data):
        if ticker_data is None or ticker_data.empty or len(ticker_data) < 3:
            return -1, -1
        highest_swing = -1
        lowest_swing = -1
        for i in range(1, ticker_data.shape[0]-1):
            if ticker_data['High'].iloc[i] > ticker_data['High'].iloc[i-1] and ticker_data['High'].iloc[i] > ticker_data['High'].iloc[i+1]:
                if highest_swing == -1 or ticker_data['High'].iloc[i] > ticker_data['High'].iloc[highest_swing]:
                    highest_swing = i
            if ticker_data['Low'].iloc[i] < ticker_data['Low'].iloc[i-1] and ticker_data['Low'].iloc[i] < ticker_data['Low'].iloc[i+1]:
                if lowest_swing == -1 or ticker_data['Low'].iloc[i] < ticker_data['Low'].iloc[lowest_swing]:
                    lowest_swing = i
        return highest_swing, lowest_swing

    def calculate_fibonacci_levels(self, ticker_data, highest_swing, lowest_swing):
        ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        colors = ["black", "r", "g", "b", "cyan", "magenta", "yellow"]
        levels = []
        if highest_swing == -1 or lowest_swing == -1:
            return levels, colors, ratios
        max_level = ticker_data['High'].iloc[highest_swing]
        min_level = ticker_data['Low'].iloc[lowest_swing]
        for ratio in ratios:
            if highest_swing > lowest_swing:
                levels.append(max_level - (max_level - min_level) * ratio)
            else:
                levels.append(min_level + (max_level - min_level) * ratio)
        return levels, colors, ratios

    def plot_chart_fibonacci_retracement_html(self):
        ticker_data = self.get_history_data()
        if ticker_data is None or ticker_data.empty or len(ticker_data) < 3:
            return "<p>Insufficient data for Fibonacci Retracement chart</p>"
        highest_swing, lowest_swing = self.calculate_swing(ticker_data)
        levels, colors, ratios = self.calculate_fibonacci_levels(ticker_data, highest_swing, lowest_swing)
        if not levels:
            return "<p>No valid swing points for Fibonacci Retracement</p>"

        def plot_func():
            plt.rc('font', size=14)
            plt.plot(ticker_data['Close'])
            for i in range(len(levels)):
                plt.hlines(levels[i], xmin=ticker_data.index[0], xmax=ticker_data.index[-1], label="{:.1f}%".format(ratios[i] * 100), colors=colors[i], linestyles="dashed")

        return self._plot_to_html(plot_func, f"[DA] {self.name_ticker.upper()} Stock Data ({self.data_period}) with Fibonacci Retracement Levels")

class OBV(Stock):
    def __init__(self, name_ticker, data_period='1y'):
        super().__init__(name_ticker)
        self.set_data_period(data_period)
        self.chart_figsize = (20, 12)

    def calculate_obv(self):
        ticker_data = self.get_history_data()
        if ticker_data is None or ticker_data.empty or len(ticker_data) < 2:
            return pd.Series()
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

    def plot_chart_obv_html(self):
        obv_data = self.calculate_obv()
        if obv_data.empty:
            return "<p>Insufficient data for OBV chart</p>"

        def plot_func():
            plt.plot(obv_data.index, obv_data, label='OBV')
            plt.axhline(y=0, color='black', linestyle='--')

        return self._plot_to_html(plot_func, f"[DA] OBV of symbol: {self.name_ticker} (Period: {self.data_period})")

class AccumulationDistributionLine(Stock):
    def __init__(self, name_ticker, data_period='1y'):
        super().__init__(name_ticker)
        self.set_data_period(data_period)
        self.chart_figsize = (20, 12)

    def calculate_adl(self):
        df = self.get_history_data()
        if df is None or df.empty or len(df) < 2:
            return pd.DataFrame()
        df['CMF Multiplier'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
        df['CMF Volume'] = df['CMF Multiplier'] * df['Volume']
        df['ADL'] = df['CMF Volume'].cumsum()
        return df[['ADL']]

    def plot_chart_ADL_html(self):
        adl_data = self.calculate_adl()
        if adl_data.empty:
            return "<p>Insufficient data for ADL chart</p>"

        def plot_func():
            plt.plot(adl_data.index, adl_data['ADL'], label='ADL')

        return self._plot_to_html(plot_func, f"[DA] ADL of symbol: {self.name_ticker} (Period: {self.data_period})")

class Analysis_TA(BollingerBands, MovingAverage, ADX, VWAP, StochasticOscillator, RSI, MADC, FibonacciRetracement, OBV, AccumulationDistributionLine):
    def __init__(self, name_ticker, data_period='1y'):
        super().__init__(name_ticker, data_period)

class SharpeRatio(Stock):
    def __init__(self, name_ticker, data_period='1y', risk_free_rate=0.05):
        super().__init__(name_ticker)
        self.set_data_period(data_period)
        self.risk_free_rate = risk_free_rate
        self.stock_data = self.get_history_data()
        self.daily_returns = self.stock_data['Close'].pct_change().dropna() if self.stock_data is not None else pd.Series()
        self.annual_returns = self.daily_returns.mean() * 252 if not self.daily_returns.empty else 0
        self.annual_volatility = self.daily_returns.std() * np.sqrt(252) if not self.daily_returns.empty else 0
        self.sharpe_ratio = (self.annual_returns - self.risk_free_rate) / (self.annual_volatility + 1e-10)
        self.chart_figsize = (20, 12)

    def plot_returns(self):
        if self.stock_data is None or self.stock_data.empty:
            return "<p>No data available for Returns chart</p>"
        cumulative_returns = (self.stock_data['Close'] / self.stock_data['Close'].iloc[0] - 1) * 100
        def plot_func():
            plt.plot(cumulative_returns)
            plt.ylabel("Cumulative Returns (%)")
        return self._plot_to_html(plot_func, f"Cumulative Returns of {self.name_ticker.upper()} ({self.data_period})")

class RRnRRR(Stock):
    def __init__(self, name_ticker):
        super().__init__(name_ticker)
        self.data = self.get_history_data()

    def get_rr(self, entry_price, stop_loss):
        if self.data is None or self.data.empty:
            return 0
        reward = abs(self.data['Close'].iloc[-1] - entry_price)
        risk = abs(entry_price - stop_loss)
        return reward / (risk + 1e-10)

    def get_rrr(self, entry_price, stop_loss, target_price):
        if self.data is None or self.data.empty:
            return 0
        reward = abs(target_price - entry_price)
        risk = abs(entry_price - stop_loss)
        return reward / (risk + 1e-10)

class BullBearIndicator(Stock):
    def __init__(self, name_ticker: str, data_period: str = '1y', stock_data: Optional[pd.DataFrame] = None):
        """
        Initialize BullBearIndicator with stock data and technical analysis parameters.
        
        Args:
            name_ticker: Stock ticker symbol
            data_period: Period for historical data (default '1y')
            stock_data: Preloaded stock data (optional)
        """
        super().__init__(name_ticker)
        self.set_data_period(data_period)
        self.stock_data = self._validate_stock_data(stock_data)
        
        # Technical indicator parameters
        self.short_window = 20
        self.long_window = 50
        self.tf_rsi = 14
        self.std_dev = 2
        self.tf_bb = 20
        
        # Cache for calculated indicators
        self._indicators: Dict[str, Union[pd.Series, float]] = {}
        self._data_validated = False

    def _validate_stock_data(self, stock_data: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Validate and clean the stock data."""
        if stock_data is None:
            stock_data = self.get_history_data()
        
        if stock_data is not None:
            if 'Close' not in stock_data.columns:
                logging.error(f"Stock data for {self.name_ticker} missing 'Close' column")
                return None
            
            stock_data = stock_data.dropna()
            if len(stock_data) < max(self.long_window, self.tf_bb, self.tf_rsi):
                logging.warning(f"Insufficient data points for {self.name_ticker}")
                return None
            
            # Ensure data is sorted chronologically
            stock_data = stock_data.sort_index(ascending=True)
            
        return stock_data

    def _ensure_data_validation(self) -> bool:
        """Check if we have valid data for calculations."""
        if self.stock_data is None or self.stock_data.empty:
            return False
        
        if not self._data_validated:
            self.stock_data = self._validate_stock_data(self.stock_data)
            self._data_validated = True
            
        return self.stock_data is not None

    def calculate_moving_averages(self) -> None:
        """Calculate and cache short and long moving averages."""
        if not self._ensure_data_validation():
            return
            
        if 'SMA_short' not in self._indicators:
            self._indicators['SMA_short'] = self.stock_data['Close'].rolling(window=self.short_window).mean()
            self._indicators['SMA_long'] = self.stock_data['Close'].rolling(window=self.long_window).mean()

    def is_bullish_ma(self) -> bool:
        """Check for bullish moving average crossover."""
        if not self._ensure_data_validation():
            return False
            
        self.calculate_moving_averages()
        last_price = self.stock_data['Close'].iloc[-1]
        sma_short = self._indicators['SMA_short'].iloc[-1]
        sma_long = self._indicators['SMA_long'].iloc[-1]
        
        return (pd.notna(sma_short) and pd.notna(sma_long) and 
                sma_short > sma_long and last_price > sma_short)

    def is_bearish_ma(self) -> bool:
        """Check for bearish moving average crossover."""
        if not self._ensure_data_validation():
            return False
            
        self.calculate_moving_averages()
        last_price = self.stock_data['Close'].iloc[-1]
        sma_short = self._indicators['SMA_short'].iloc[-1]
        sma_long = self._indicators['SMA_long'].iloc[-1]
        
        return (pd.notna(sma_short) and pd.notna(sma_long) and 
                sma_short < sma_long and last_price < sma_short)

    def calculate_rsi(self) -> Optional[pd.Series]:
        """Calculate and cache RSI values."""
        if not self._ensure_data_validation():
            return None
            
        if f'RSI_{self.tf_rsi}' not in self._indicators:
            delta = self.stock_data['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.rolling(self.tf_rsi).mean()
            avg_loss = loss.rolling(self.tf_rsi).mean()
            
            rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
            self._indicators[f'RSI_{self.tf_rsi}'] = 100 - (100 / (1 + rs))
            
        return self._indicators[f'RSI_{self.tf_rsi}']

    def is_bullish_rsi(self) -> bool:
        """Check for bullish RSI signal (oversold condition)."""
        rsi = self.calculate_rsi()
        if rsi is None or len(rsi) < 2:
            return False
            
        return (pd.notna(rsi.iloc[-1]) and pd.notna(rsi.iloc[-2]) and 
                rsi.iloc[-1] < 30 and rsi.iloc[-2] >= 30)

    def is_bearish_rsi(self) -> bool:
        """Check for bearish RSI signal (overbought condition)."""
        rsi = self.calculate_rsi()
        if rsi is None or len(rsi) < 2:
            return False
            
        return (pd.notna(rsi.iloc[-1]) and pd.notna(rsi.iloc[-2]) and 
                rsi.iloc[-1] > 70 and rsi.iloc[-2] <= 70)

    def calculate_bollinger_bands(self) -> None:
        """Calculate and cache Bollinger Bands."""
        if not self._ensure_data_validation():
            return
            
        if 'BB_MA' not in self._indicators:
            self._indicators['BB_MA'] = self.stock_data['Close'].rolling(window=self.tf_bb).mean()
            self._indicators['BB_STD'] = self.stock_data['Close'].rolling(window=self.tf_bb).std()
            self._indicators['BB_Upper'] = self._indicators['BB_MA'] + (self._indicators['BB_STD'] * self.std_dev)
            self._indicators['BB_Lower'] = self._indicators['BB_MA'] - (self._indicators['BB_STD'] * self.std_dev)

    def is_bullish_bollinger_bands(self) -> bool:
        """Check for bullish Bollinger Band signal (price below lower band)."""
        if not self._ensure_data_validation():
            return False
            
        self.calculate_bollinger_bands()
        last_close = self.stock_data['Close'].iloc[-1]
        lower_band = self._indicators['BB_Lower'].iloc[-1]
        
        return pd.notna(lower_band) and last_close < lower_band

    def is_bearish_bollinger_bands(self) -> bool:
        """Check for bearish Bollinger Band signal (price above upper band)."""
        if not self._ensure_data_validation():
            return False
            
        self.calculate_bollinger_bands()
        last_close = self.stock_data['Close'].iloc[-1]
        upper_band = self._indicators['BB_Upper'].iloc[-1]
        
        return pd.notna(upper_band) and last_close > upper_band

def check_bullish(name_ticker, no_signal=2, data_period='6mo', bb_signal=None):
    if bb_signal is None:
        bb_signal = BullBearIndicator(name_ticker, data_period)
    if bb_signal.stock_data is None:
        return None
    indicators = [
        {'name': 'Moving Average', 'is_bullish': bb_signal.is_bullish_ma()},
        {'name': 'Breakdown', 'is_bullish': bb_signal.is_bullish_bd()},
        {'name': 'OversoldSignal(RSI)', 'is_bullish': bb_signal.is_bullish_rsi()},
        {'name': 'Bollinger Bands', 'is_bullish': bb_signal.is_bullish_bollinger_bands()}
    ]
    tickers = [name_ticker for i in indicators if i['is_bullish']]
    return name_ticker if len(tickers) >= no_signal else None

def check_bearish(name_ticker, no_signal=2, data_period='6mo', bb_signal=None):
    if bb_signal is None:
        bb_signal = BullBearIndicator(name_ticker, data_period)
    if bb_signal.stock_data is None:
        return None
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
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        dfs = pd.read_html(url, header=0)
        sp500_tickers = dfs[0]['Symbol'].tolist()
        sp500_tickers = [ticker.replace(".", "-") for ticker in sp500_tickers]
        data = yf.download(sp500_tickers, period='6mo', group_by='ticker', progress=False, auto_adjust=True)
        for ticker in sp500_tickers:
            if ticker in data.columns.get_level_values(0):
                ticker_data = data[ticker].dropna()
                if not ticker_data.empty:
                    bb_signal = BullBearIndicator(ticker, '6mo', stock_data=ticker_data)
                    if check_bullish(ticker, bb_signal=bb_signal):
                        lst_bullish.append(ticker)
                    if check_bearish(ticker, bb_signal=bb_signal):
                        lst_bearish.append(ticker)
    except Exception as e:
        logging.error(f"Error scanning S&P 500: {e}")
    return lst_bullish, lst_bearish

def generate_html_header():
    today_date = datetime.now().strftime("%b, %d %Y")
    return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Stock Analysis Report at {today_date}</title>
            <style>
                .tab {{
                    overflow: hidden;
                    border: 1px solid #ccc;
                    background-color: #f1f1f1;
                }}
                .tab button {{
                    background-color: inherit;
                    float: left;
                    border: none;
                    outline: none;
                    cursor: pointer;
                    padding: 14px 16px;
                    transition: 0.3s;
                }}
                .tab button:hover {{
                    background-color: #ddd;
                }}
                .tab button.active {{
                    background-color: #ccc;
                }}
                .tabcontent {{
                    display: none;
                    padding: 6px 12px;
                    border: 1px solid #ccc;
                    border-top: none;
                }}
                .chart-container {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                }}
                .chart-img {{
                    width: 25%;
                    height: auto;
                    margin: 5px;
                }}
                .bordered-table {{
                    border: 1px solid black;
                    border-collapse: collapse;
                    width: 30%;
                }}
                th, td {{
                    border: 1px solid black;
                    padding: 8px;
                    text-align: left;
                }}
                ul.menu {{
                    list-style-type: none;
                    margin: 0;
                    padding: 0;
                    background-color: #f1f1f1;
                }}
                ul.menu li {{
                    display: inline-block;
                }}
                ul.menu li a {{
                    display: block;
                    padding: 8px 16px;
                    text-decoration: none;
                    color: #333;
                }}
                ul.menu li a:hover {{
                    background-color: #ddd;
                }}
            </style>
        </head>
        <body>
        <h1>[DEMO] Stock Analysis Report</h1>
        <p>Generated by Market_Analysis_App at {today_date} <a href="https://github.com/gilbertycc/myCodes/tree/main/Market_Analysis_App">Link to the Market Analysis App repository</a></p>
        <div class="tab">
    """

def generate_html_body_stock(name_stock, review_stock, bb_signals_html_output, index_stock):
    return f"""
        <div id="stock{index_stock}" class="tabcontent">
            <section id="stockInfo">
                <h3>Stock Info</h3>
                {review_stock.get_stock_info_html()}
            </section>
            <section id="marketSignal">
                <h3>Market Signal</h3>
                {bb_signals_html_output}
            </section>
            <section id="technicalAnalysisChart">
                <h3>Technical Analysis Chart</h3>
            </section>
            <div class="chart-container">
                {review_stock.plot_chart_MovingAverage_html()}
                {review_stock.plot_chart_BollingerBands_html()}
                {review_stock.plot_chart_ADX_html()}
                {review_stock.plot_chart_vwap_html()}
                {review_stock.plot_chart_stochastic_oscillator_html()}
                {review_stock.plot_chart_rsi_html()}
                {review_stock.plot_chart_madc_html()}
                {review_stock.plot_chart_fibonacci_retracement_html()}
                {review_stock.plot_chart_obv_html()}
                {review_stock.plot_chart_ADL_html()}
            </div>
            <section id="machineLearningChart">
                <h3>Machine Learning Analysis Table</h3>
            </section>
            <table class="bordered-table">
                <tr><td>Symbol</td><td>Prediction Date</td><td>Model</td><td>Predicted Close</td><td>Actual Close</td><td>R2</td><td>MSE</td><td>MAPE</td></tr>
                {review_stock.ml_RFR_html()}
            </table>
        </div>
    """

def generate_html_body_tab(name_stock, index_stock):
    return f"""
        <button class="tablinks" onclick="openStock(event, 'stock{index_stock}')" {'id="defaultOpen"' if index_stock == 0 else ''}>{name_stock}</button>
    """

def generate_html_end():
    return f"""
        <div class="disclaimer">
            <h3>Disclaimer</h3>
            <p>The data analysis and machine learning stock price analysis tool provided in this Github repository are for informational and educational purposes only. The tool does not constitute financial advice, nor should it be relied upon as a substitute for professional financial advice.</p>
            <p>The data and predictions provided by the tool may not be accurate, complete, or up-to-date. We do not guarantee the accuracy, timeliness, completeness, or usefulness of any information provided by the tool.</p>
            <p>We are not responsible for any losses or damages that may arise from the use of this tool or reliance on any information provided by the tool.</p>
            <p>Users of the tool should conduct their own research and seek professional financial advice before making any investment decisions. By using this tool, you acknowledge and agree to the terms of this disclaimer.</p>
        </div>
        <script>
            function openStock(evt, stockName) {{
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {{
                    tabcontent[i].style.display = "none";
                }}
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {{
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }}
                document.getElementById(stockName).style.display = "block";
                evt.currentTarget.className += " active";
            }}
            var defaultTab = document.getElementById("defaultOpen");
            if (defaultTab) {{ defaultTab.click(); }}
            function enlargeImage(img) {{
                img.style.width = "85%";
                img.style.height = "85%";
 PRIVMSG #xai-alignment :img.style.width = "25%";
                img.style.height = "auto";
            }}
        </script>
        </body>
    </html>
    """

def add_html_menu():
    return f"""
        <nav>
            <ul class="menu">
                <li><a href="#stockInfo">Stock Info</a></li>
                <li><a href="#marketSignal">Market Signal</a></li>
                <li><a href="#technicalAnalysisChart">Technical Analysis Chart</a></li>
                <li><a href="#machineLearningChart">Machine Learning Analysis Chart</a></li>
            </ul>
        </nav>
    """

def generate_bb_signals_html(bb_signal, ticker):
    GREEN = "#008000"
    RED = "#FF0000"
    ENDC = "</span>"
    bb_signals = {
        'MA': bb_signal.is_bullish_ma,
        'Breakdown': bb_signal.is_bullish_bd,
        'OversoldSignal(RSI)': bb_signal.is_bullish_rsi,
        'Bollinger Bands': bb_signal.is_bullish_bollinger_bands
    }
    bb_signals_html = f"""
        <table class="bordered-table">
        <tr><td>Signal Type</td><td>Signal</td></tr>
    """
    for signal_type, signal_func in bb_signals.items():
        signal = 'Bullish signal detected' if signal_func() else 'Bearish signal detected' if bb_signal.is_bearish_ma() else 'No clear signal detected'
        color = GREEN if signal_func() else RED if bb_signal.is_bearish_ma() else ENDC
        bb_signals_html += f"<tr><td>{signal_type}</td><td><span style='color:{color}'>{signal}</span></td></tr>"
    bb_signals_html += "</table>"
    return bb_signals_html

def generate_full_html_report(stocks, export_file_path='./'):
    export_full_path = f"{export_file_path}index.html"
    with open(export_full_path, 'w') as f_html:
        f_html.write(generate_html_header())
        for i, stock in enumerate(stocks):
            f_html.write(generate_html_body_tab(stock, i))
        f_html.write("</div>")
        f_html.write(add_html_menu())
        for i, stock in enumerate(stocks):
            review_stock = Analysis_TA(stock, '1y')
            bb_signal = BullBearIndicator(stock, '6mo')
            bb_signals_html = generate_bb_signals_html(bb_signal, stock)
            f_html.write(generate_html_body_stock(stock, review_stock, bb_signals_html, i))
        f_html.write(generate_html_end())

def main():
    debug_mode = True  # Set to False to scan all S&P 500 tickers
    shorted_list_stock = ['MSFT', 'GOOG'] if debug_mode else scan_sp500_bb()[0] + scan_sp500_bb()[1]
    if not shorted_list_stock:
        logging.warning("No stocks found for analysis")
        return
    generate_full_html_report(shorted_list_stock)

if __name__ == "__main__":
    main()
