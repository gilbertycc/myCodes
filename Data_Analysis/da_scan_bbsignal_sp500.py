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



class BullBearIndicator(Stock):

    def __init__(self, name_ticker, data_period='1y'):
        Stock.__init__(self, name_ticker)
        self.data_period = data_period
        # self.stock_data = self.get_history_data().dropna()
        history_data = self.get_history_data()
        if history_data is not None:
          self.stock_data = history_data.dropna()
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



def print_stock_bbsignal(name_ticker, data_period='6mo'):
      
    GREEN = '\033[92m'
    RED = '\033[91m'
    ENDC = '\033[0m'

    bb_signal = BullBearIndicator(name_ticker, data_period)
    print (bb_signal)

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

    print (lst_bullish)
    print (lst_bearish)


class VolumnIndicator(Stock):
    
    def __init__(self, name_ticker, data_period='1y'):
        Stock.__init__(self, name_ticker)
        self.data_period = self.data_period
        self.df = self.get_history_data()
        self.df['Typical Price'] = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        self.df['Money Flow'] = self.df['Typical Price'] * self.df['Volume']
        self.df['Direction'] = 0
        self.df.loc[self.df['Typical Price'] > self.df['Typical Price'].shift(), 'Direction'] = 1
        self.df.loc[self.df['Typical Price'] < self.df['Typical Price'].shift(), 'Direction'] = -1
    
    ''' Money Flow Index (MFI) '''
    def calculate_mfi(self, n=14):
        self.df['Raw Money Flow'] = self.df['Direction'] * self.df['Money Flow']
        self.df['Positive Money Flow'] = 0
        self.df.loc[self.df['Raw Money Flow'] > 0, 'Positive Money Flow'] = self.df['Raw Money Flow']
        self.df['Negative Money Flow'] = 0
        self.df.loc[self.df['Raw Money Flow'] < 0, 'Negative Money Flow'] = abs(self.df['Raw Money Flow'])
        self.df['Positive Money Flow Sum'] = self.df['Positive Money Flow'].rolling(window=n).sum()
        self.df['Negative Money Flow Sum'] = self.df['Negative Money Flow'].rolling(window=n).sum()
        self.df['Money Ratio'] = self.df['Positive Money Flow Sum'] / self.df['Negative Money Flow Sum']
        self.df['MFI'] = 100 - (100 / (1 + self.df['Money Ratio']))
        return self.df[['MFI']]
    
    def get_overbought_oversold_mfi(self, threshold=80):
        mfi = self.calculate_mfi().copy()
        mfi.loc[:, 'Signal'] = 'Neutral'
        mfi.loc[mfi['MFI'] > threshold, 'Signal'] = 'Overbought'
        mfi.loc[mfi['MFI'] < 100-threshold, 'Signal'] = 'Oversold'
        return mfi

    ''' Volume Weighted Average Price (VWAP) '''
    def calculate_vwap(self, n=20):
        self.df['TP'] = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        self.df['TPV'] = self.df['TP'] * self.df['Volume']
        self.df['Cumulative TPV'] = self.df['TPV'].cumsum()
        self.df['Cumulative Volume'] = self.df['Volume'].cumsum()
        self.df['VWAP'] = self.df['Cumulative TPV'] / self.df['Cumulative Volume']

        return self.df
    
    def get_overbought_oversold_vwap(self, threshold=0.1):
        vwap = self.calculate_vwap().copy()
        vwap['Signal'] = 'Neutral'
        vwap['Return'] = vwap['Close'].pct_change()
        vwap.loc[vwap['Return'] > threshold, 'Signal'] = 'Overbought'
        vwap.loc[vwap['Return'] < -threshold, 'Signal'] = 'Oversold'
        return vwap[['VWAP', 'Signal']]

    ''' Chaikin Money Flow (CMF) '''
    def calculate_cmf(self):
        money_flow_multiplier = ((self.df['Close'] - self.df['Low']) - (self.df['High'] - self.df['Close'])) / (self.df['High'] - self.df['Low'])
        money_flow_volume = money_flow_multiplier * self.df['Volume']
        cmf = money_flow_volume.rolling(20).sum() / self.df['Volume'].rolling(20).sum()
        return cmf

    def get_overbought_oversold_cmf(self, threshold=0.25):
        self.df = self.calculate_cmf().to_frame().rename(columns={0: 'CMF'})
        self.df['Signal'] = 'Neutral'
        self.df.loc[self.df['CMF'] >= threshold, 'Signal'] = 'Overbought'
        self.df.loc[self.df['CMF'] <= -threshold, 'Signal'] = 'Oversold'
        return self.df[['CMF', 'Signal']]

    def calculate_obv(self):
        self.df['Daily Return'] = self.df['Adj Close'].pct_change()
        self.df['Direction'] = self.df['Daily Return'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        self.df['Volume Signed'] = self.df['Volume'] * self.df['Direction']
        self.df['OBV'] = self.df['Volume Signed'].cumsum()
        return self.df['OBV']
    
    def get_overbought_oversold_obv(self, threshold=0):
        obv = self.calculate_obv()
        obv = self.df.copy()
        obv['Signal'] = 'Neutral'
        obv['Return'] = obv['Adj Close'].pct_change()
        obv.loc[obv['OBV'] > threshold, 'Signal'] = 'Overbought'
        obv.loc[obv['OBV'] < -threshold, 'Signal'] = 'Oversold'
        return obv[['OBV', 'Signal']]



##### MAIN #####

# scan_sp500_bb()


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
    '''
    # Money Flow Index (MFI)
    #mfi = VolumnIndicator(ticker, data_period='1mo')
    #mfi_df = mfi.get_overbought_oversold_mfi()
    # print(mfi_df.tail(1).values.squeeze())
    #if mfi_df.tail(1)['Signal'].values != 'Neutral': 
    #    print (ticker, mfi_df.tail(1)['Signal'].values)
        # Volume Weighted Average Price (VWAP)
    vp = VolumnIndicator(ticker, data_period='1mo')
    vp_df = vp.get_overbought_oversold_vwap()
    print (vp.get_overbought_oversold_vwap().tail(1).values.squeeze())
    if vp_df.tail(1)['Signal'].values != 'Neutral':
        print (ticker, vp_df.tail(1)['Signal'].values)
    
    cmf = VolumnIndicator(ticker, data_period='1mo')
    cmf_df = cmf.get_overbought_oversold_cmf()
    if cmf_df.tail(1)['Signal'].values != 'Neutral':
        print (ticker, cmf_df.tail(1)['Signal'].values)
    '''
        
    obv = VolumnIndicator(ticker, data_period='1mo')
    obv_df = obv.get_overbought_oversold_obv()
    if obv_df.tail(1)['Signal'].values != 'Neutral':
        print (ticker, obv_df.tail(1)['Signal'].values)
'''
Some common volume indicators include:

On-Balance Volume (OBV)
Chaikin Money Flow (CMF)
Volume Weighted Average Price (VWAP)
Money Flow Index (MFI)
'''

