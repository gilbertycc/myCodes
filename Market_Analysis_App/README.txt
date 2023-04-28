### Disclaimer 

The data analysis and machine learning stock price analysis tool provided in this Github repository are for informational and educational purposes only. The tool does not constitute financial advice, nor should it be relied upon as a substitute for professional financial advice.

The data and predictions provided by the tool may not be accurate, complete, or up-to-date. We do not guarantee the accuracy, timeliness, completeness, or usefulness of any information provided by the tool.

We are not responsible for any ``losses or damages that may arise from the use of this tool or reliance on any information provided by the tool.

Users of the tool should conduct their own research and seek professional financial advice before making any investment decisions. By using this tool, you acknowledge and agree to the terms of this disclaimer.

###

Demo Report Link: https://gilbertycc.github.io/myCodes/

This is a Python application that analyzes stocks from the S&P500 stock list and generates an analysis report in HTML format.

Required Libraries
The following libraries are required to run the application:
# pip install yfinance pandas numpy matplotlib ipython yfinance

datetime
timedelta
yfinance
pandas
numpy
matplotlib.pyplot
io.StringIO
io.BytesIO
base64.b64encode
IPython.core.display.HTML (ipython==7.16.1)

Usage
Clone the repository to your local machine.
Open the command line or terminal and navigate to the project directory.
Run the main.py script using the command python main.py.
Follow the instructions provided by the application to analyze the S&P500 stocks and generate an analysis report.
Functionality of the App
The Market Analysis App has the following functionality:

Scans the S&P500 stock list with a specified model and shortlists the stocks at end of day (EOD).
Generates an analysis report in HTML format that provides detailed information on the shortlisted stocks.
The analysis report includes the following information for each stock:

Symbol Type
Ticker
Current Data Period
Latest Price
Market Signal (including Bullish & Bearish Signal - MA, Breakdown, OversoldSignal(RSI), and Bollinger Bands)
Technical Chart
The analysis report is saved as an HTML file that can be viewed in a web browser. The HTML report can also be customized and styled as needed.

Note: This application requires an internet connection to download the stock data.