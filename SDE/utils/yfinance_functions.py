import yfinance as yf  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from datetime import datetime, timedelta


def plot_stock_curves(historical_data, title, ylabel="price"):
    colors = plt.cm.Set2

    # Drop the 'Symbol' column if present
    if 'Symbol' in historical_data.columns:
        historical_data = historical_data.drop(columns='Symbol')

    # Plot the curves for each underlying
    plt.figure(figsize=(20, 6))
    for i, column in enumerate(historical_data.columns):
        plt.plot(historical_data.index, historical_data[column], label=column, color=colors(i))

    plt.title('Historical Stock Prices')
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(False)
    plt.show()


def compute_annual_volatility(historical_series):
    """
    Compute historical annualized volatility for a given DataFrame of stock prices.
    
    -> assume dataframe 
    historical_data (pandas.DataFrame): DataFrame containing historical stock prices.
    Returns:  Historical annualized volatility.
    """

    # Calculate daily returns
    historical_returns = historical_series.pct_change()

    # Calculate daily volatility
    daily_volatility = historical_returns.std()

    # Annualize volatility
    trading_days_per_year = 252  # Assuming 252 trading days in a year
    annual_volatility = daily_volatility * (trading_days_per_year ** 0.5)

    return annual_volatility


def compute_correlation_matrix(historical_data):
    """
    Calculate the correlation between two stocks based on their historical daily returns.

    Returns: Correlation coefficient between the two stocks.
    """
    correlation_matrix = historical_data.corr()
    return correlation_matrix


def get_historical_data(ticker_symbols, start_date=None, end_date=None, period=None):
    """
    Get historical data for a list of ticker symbols over a specified period.
    
    ticker_symbols (list): List of ticker symbols (strings).
    period (str): Period for historical data ('1d','1mo','1y').
    Returns: pandas.DataFrame: DataFrame containing historical data for each ticker symbol.
    """

    def shift_date_by_one_day(date_str):
        # Parse the input date string into a datetime object
        date = datetime.strptime(date_str, '%Y-%m-%d')
        # Shift the date by one day
        shifted_date = date + timedelta(days=1)
        # Convert the shifted date back to a string in the desired format
        shifted_date_str = shifted_date.strftime('%Y-%m-%d')
        return shifted_date_str

    data = pd.DataFrame()

    if not period: assert start_date and end_date, "if no period is given, must give start and end dates"

    for symbol in ticker_symbols:
        # Create a Ticker object
        ticker = yf.Ticker(symbol)

        # Get historical market data between start_date and end_date
        historical_data = ticker.history(start=start_date, end=end_date)

        # Extract the 'Close' column and rename it to the symbol
        historical_close = historical_data['Close'].to_frame().rename(columns={'Close': symbol})

        # Concatenate the historical closing prices
        data = pd.concat([data, historical_close], axis=1)

    # retreat indices
    data.index = pd.to_datetime(data.index).normalize()
    return data


def get_date_data(ticker_symbols, at_date):
    """
    Get historical data for a list of ticker symbols over a specified period.
    
    ticker_symbols (list): List of ticker symbols (strings).
    at_date (str): if at_date is given than returns only value at that date
    Returns: pandas.DataFrame: DataFrame containing historical data at the specific date
    """

    def shift_date_by_one_day(date_str):
        # Parse the input date string into a datetime object
        date = datetime.strptime(date_str, '%Y-%m-%d')
        # Shift the date by one day
        shifted_date = date + timedelta(days=1)
        # Convert the shifted date back to a string in the desired format
        shifted_date_str = shifted_date.strftime('%Y-%m-%d')
        return shifted_date_str

    next_day = shift_date_by_one_day(at_date)
    return get_historical_data(ticker_symbols, start_date=at_date, end_date=next_day)
