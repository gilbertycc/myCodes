import importlib.util
import subprocess
import sys

# Check and install squarify if not present
if importlib.util.find_spec("squarify") is None:
    print("[Debug] *****squarify not found. Installing squarify...*****")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "squarify"])
else:
    print("[Debug] *****squarify is already installed.*****")

# Imports
import numpy as np
import yfinance as yf
import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from io import StringIO
import squarify
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import holidays

# Settings
pd.options.mode.chained_assignment = None
np.set_printoptions(legacy='1.25')

def get_sp500_list():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        response = requests.get(url)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        sp500_df = tables[0]
        sp500_df = sp500_df[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry', 'Headquarters Location', 'Date added', 'CIK']]
        sp500_df['Symbol'] = sp500_df['Symbol'].str.replace('.', '-', regex=False)
        return sp500_df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    except ValueError as e:
        print(f"Error parsing table: {e}")
        return None

def get_closing_price_and_market_cap(ticker, date=(datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d'), max_attempts=5):
    try:
        date_obj = pd.to_datetime(date)
        us_holidays = holidays.US(years=date_obj.year)
        for attempt in range(max_attempts):
            if date_obj in us_holidays or date_obj.weekday() >= 5:
                date_obj -= timedelta(days=1)
                continue
            next_day = date_obj + pd.Timedelta(days=1)
            stock = yf.Ticker(ticker)
            hist = stock.history(start=date_obj.strftime('%Y-%m-%d'), end=next_day.strftime('%Y-%m-%d'))
            if not hist.empty:
                closing_price = hist['Close'].values[0]
                info = stock.info
                market_cap = info.get('marketCap', 'N/A')
                return [ticker, date_obj.strftime('%Y-%m-%d'), closing_price, market_cap]
            date_obj -= timedelta(days=1)
        return f"No data available for {ticker} on {date} after {max_attempts} attempts (market may have been closed)."
    except Exception as e:
        return f"An error occurred for {ticker}: {e}"

def get_dates():
    us_holidays = holidays.US(years=datetime.today().year)
    today = datetime.today() - timedelta(days=1)
    date1 = today
    while date1.weekday() >= 5 or date1 in us_holidays:
        date1 -= timedelta(days=1)
    date2 = date1 - timedelta(days=1)
    while date2.weekday() >= 5 or date2 in us_holidays:
        date2 -= timedelta(days=1)
    return date1.strftime('%Y-%m-%d'), date2.strftime('%Y-%m-%d')

# Main execution
td, ld = get_dates()
sp500_df = get_sp500_list()

tickers = []
current_dates = []
current_closes = []
previous_closes = []
market_caps = []

for t in sp500_df['Symbol'].values:
    prev_data = get_closing_price_and_market_cap(t, ld)
    curr_data = get_closing_price_and_market_cap(t, td)
    if isinstance(prev_data, list) and isinstance(curr_data, list):
        tickers.append(t)
        current_dates.append(td)
        current_closes.append(curr_data[2])
        previous_closes.append(prev_data[2])
        market_caps.append(curr_data[3])

result_df = pd.DataFrame({
    'Ticker': tickers,
    'Current_Date': current_dates,
    'Current_Close': current_closes,
    'Previous_Close': previous_closes,
    'Market_Cap': market_caps
})

result_df['Price_Diff'] = result_df['Current_Close'] - result_df['Previous_Close']
result_df['Price_Change_%'] = ((result_df['Price_Diff'] / result_df['Previous_Close']) * 100).round(2)
result_df = result_df.sort_values(by='Market_Cap', ascending=False)

# Filter top 100 stocks by Market_Cap
top_df = result_df.nlargest(100, 'Market_Cap')
top_df['Market_Cap'] = top_df['Market_Cap'].clip(lower=1)

sizes = top_df['Market_Cap'].values
labels = top_df.apply(
    lambda x: f"{x['Ticker']}\n{x['Price_Diff']:+.2f}\n{x['Price_Change_%']:+.2f}%",
    axis=1
)

# Enhanced color scheme
colors = []
for change in top_df['Price_Change_%']:
    if change >= 5:
        colors.append('#006400')  # Deep green
    elif change <= -5:
        colors.append('#8B0000')  # Deep red
    elif change >= 0:
        colors.append('#2ca02c')  # Standard green
    else:
        colors.append('#d62728')  # Standard red

# Calculate font sizes
min_font, max_font = 8, 16
norm_sizes = (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-10)
font_sizes = min_font + (max_font - min_font) * norm_sizes

# Create the treemap
plt.figure(figsize=(16, 9), facecolor='#f5f5f5')
squarify.plot(
    sizes=sizes,
    color=colors,
    alpha=0.7,
    edgecolor='white',
    linewidth=1.5,
    pad=False
)

# Add labels
for i, rect in enumerate(plt.gca().patches):
    x, y, width, height = rect.get_x(), rect.get_y(), rect.get_width(), rect.get_height()
    label = labels.iloc[i]
    fontsize = font_sizes[i]
    price_change = top_df['Price_Change_%'].iloc[i]
    text_color = 'white' if price_change < 0 else 'black'
    plt.text(
        x + width / 2, y + height / 2, label,
        ha='center', va='center',
        fontsize=fontsize,
        fontfamily='DejaVu Sans',
        color=text_color,
        wrap=True,
        bbox=dict(facecolor='none', edgecolor='none', pad=2)
    )

# Final plot adjustments
plt.title('S&P 500 Top 100 Stocks by Market Cap', fontsize=20, fontweight='bold', pad=20, fontfamily='DejaVu Sans')
plt.suptitle(f'Daily Price Change as of {td}', fontsize=14, fontfamily='DejaVu Sans', y=0.92)
plt.axis('off')
plt.tight_layout()
plt.show()
