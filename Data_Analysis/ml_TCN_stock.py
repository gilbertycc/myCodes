import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Define the TCN model
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, num_channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm1d(num_channels),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm1d(num_channels),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm1d(num_channels),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.BatchNorm1d(num_channels),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, output_size, kernel_size, padding=(kernel_size - 1) // 2)
        )

    def forward(self, x):
        return self.tcn(x)


# Prepare the data
def prepare_data(ticker_symbol, train_size, period='max'):
    data = yf.download(ticker_symbol, period=period, progress=False)
    prices = data['Close'].values.astype(float)
    min_price = np.min(prices)
    max_price = np.max(prices)
    prices_normalized = (prices - min_price) / (max_price - min_price)
    dates = data.index.date
    train_data_size = int(len(prices_normalized) * train_size)
    train_data = prices_normalized[:train_data_size]
    test_data = prices_normalized[train_data_size:]
    test_dates = dates[train_data_size+1:]  # Exclude the first date since it's used for input

    return train_data, test_data, min_price, max_price, test_dates


# Train the model
def train(model, train_data, num_epochs, batch_size, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = TensorDataset(train_data[:-1].unsqueeze(1), train_data[1:].unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()

    for epoch in range(num_epochs):
        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

        #if (epoch + 1) % 10 == 0:
        #    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    print('Training completed!')




def convert_to_regular_price(normalized_prices, min_price, max_price):
    regular_prices = (normalized_prices * (max_price - min_price)) + min_price
    return regular_prices


def convert_to_regular_date(times, data):
    date_range = pd.date_range(start=times[0], periods=len(data), freq='D')
    return date_range.astype(str).tolist()


# Test the model
def test(model, test_data, min_price, max_price, times):
    model.eval()
    with torch.no_grad():
        inputs = test_data[:-1].unsqueeze(1)
        targets = test_data[1:].unsqueeze(1)
        outputs = model(inputs)

    inputs = convert_to_regular_price(inputs.squeeze(1).numpy(), min_price, max_price)
    targets = convert_to_regular_price(targets.squeeze(1).numpy(), min_price, max_price)
    outputs = convert_to_regular_price(outputs.squeeze(1).numpy(), min_price, max_price)
    
    dates = convert_to_regular_date(times, test_data)

    plt.plot(dates[:-1], inputs, label='Input')
    plt.plot(dates[1:], targets, label='Target')
    plt.plot(dates[1:], outputs, label='Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(12)) 

    plt.show()

def return_sp500_list():

    # Scrape S&P 500 tickers from Wikipedia
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    dfs = pd.read_html(url, header=0)
    sp500_df = dfs[0]
    sp500_tickers = sp500_df['Symbol'].tolist()

    # Replace "." with "-"
    sp500_tickers = [ticker.replace(".", "-") for ticker in sp500_tickers]

    return sp500_tickers

# Main function
if __name__ == '__main__':
    # Parameters
    #ticker_symbol = return_sp500_list()
    #ticker_symbol = ['ABT', 'AEP', 'AMGN', 'ACGL', 'BSX', 'COO', 'EW', 'FSLR', 'HSY', 'INTC', 'ICE', 'JNJ', 'K', 'KEYS', 'MRK', 'MTD', 'TAP', 'MDLZ', 'NRG', 'PCG', 'RSG', 'RMD', 'TMUS', 'WELL', 'YUM', 'ZTS']
    ticker_symbol = 'YUM'
    data_period='max'
    train_size = 0.9  # Percentage of data to use for training
    input_size = 1
    output_size = 1
    num_channels = 64
    kernel_size = 3
    dropout = 0.2
    num_epochs = 1000
    batch_size = 64
    learning_rate = 0.0001

    # Create the TCN model
    model = TCN(input_size, output_size, num_channels, kernel_size, dropout)


    # Prepare the data
    train_data, test_data, min_price, max_price, test_dates = prepare_data(ticker_symbol, train_size, period=data_period)

    # Convert the data to PyTorch tensors
    train_data = torch.FloatTensor(train_data)
    test_data = torch.FloatTensor(test_data)

    # Reshape the data for TCN input
    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)

    # Train the model
    train(model, train_data, num_epochs, batch_size, learning_rate)

    # Test the model
    test(model, test_data, min_price, max_price, test_dates)
