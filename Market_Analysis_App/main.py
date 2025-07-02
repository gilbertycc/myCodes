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

    # ... (other methods unchanged, but update 'Adj Close' to 'Close' where applicable)

class BullBearIndicator(Stock):
    def __init__(self, name_ticker, data_period='1y', stock_data=None):
        super().__init__(name_ticker)
        self.set_data_period(data_period)
        self.stock_data = stock_data if stock_data is not None else self.get_history_data()
        if self.stock_data is not None:
            self.stock_data = self.stock_data.dropna()
        self.short_window = 20
        self.long_window = 50
        self.tf_rsi = 14
        self.std_dev = 2
        self.tf_bb = 20
        self._indicators = {}

    def calculate_moving_averages(self):
        if self.stock_data is None or 'Close' not in self.stock_data.columns:
            logging.error(f"No valid data for moving averages for {self.name_ticker}")
            return
        if 'SMA_short' not in self._indicators:
            self._indicators['SMA_short'] = self.stock_data['Close'].rolling(window=self.short_window).mean()
            self._indicators['SMA_long'] = self.stock_data['Close'].rolling(window=self.long_window).mean()

    def is_bullish_ma(self):
        if self.stock_data is None or self.stock_data.empty:
            return False
        self.calculate_moving_averages()
        last_price = self.stock_data['Close'].iloc[-1]
        sma_short = self._indicators['SMA_short'].iloc[-1]
        sma_long = self._indicators['SMA_long'].iloc[-1]
        return sma_short > sma_long and last_price > sma_short

    def is_bearish_ma(self):
        if self.stock_data is None or self.stock_data.empty:
            return False
        self.calculate_moving_averages()
        last_price = self.stock_data['Close'].iloc[-1]
        sma_short = self._indicators['SMA_short'].iloc[-1]
        sma_long = self._indicators['SMA_long'].iloc[-1]
        return sma_short < sma_long and last_price < sma_short

    def calculate_rsi(self, timeframe):
        if self.stock_data is None or 'Close' not in self.stock_data.columns:
            logging.error(f"No valid data for RSI for {self.name_ticker}")
            return pd.Series()
        if f'RSI_{timeframe}' not in self._indicators:
            delta = self.stock_data['Close'].diff().dropna()
            gains = delta.clip(lower=0)
            losses = -delta.clip(upper=0)
            avg_gain = gains.rolling(timeframe).mean().dropna()
            avg_loss = losses.rolling(timeframe).mean().dropna()
            rs = avg_gain / (avg_loss + 1e-10)
            self._indicators[f'RSI_{timeframe}'] = 100 - (100 / (1 + rs))
        return self._indicators[f'RSI_{timeframe}']

    # ... (other methods updated to use 'Close' instead of 'Adj Close')

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
        {'name': 'OversoldSignal(RSI)', is_bearish': bb_signal.is_bearish_rsi()},
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
        
        # Batch download
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

# Update other classes (e.g., MovingAverage, RSI, etc.) to use 'Close' instead of 'Adj Close'
class MovingAverage(Stock):
    def plot_chart_MovingAverage_html(self):
        data_history = self.get_history_data()
        if data_history is None or data_history.empty or 'Close' not in data_history.columns:
            return "<p>No data available for Moving Average chart</p>"

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
