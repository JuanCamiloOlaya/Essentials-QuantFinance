
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def descriptive_stats(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Calculate descriptive statistics (mean, std, min, max, etc.) for
    given tickers.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing returns data.
    tickers : list[str]
        A list of tickers to calculate statistics for.

    Returns
    -------
    pd.DataFrame
        A DataFrame with descriptive statistics for each ticker.

    Raises
    ------
    ValueError
        Ensures that only the tickers present in the DataFrame are used.
    """
    # Safety check
    valid_tickers = [t for t in tickers if t in data.columns]
    if not valid_tickers:
        raise ValueError('None of the provided tickers are found in the dataset.')

    # Calculate statistics
    stats = data[valid_tickers].describe().T
    stats['skewness'] = data[valid_tickers].skew()
    stats['kurtosis'] = data[valid_tickers].kurt()

    return stats


def ts_stationarity(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the Augmented Dickey-Fuller test to each stock in a DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Each column is a stock's returns or prices.

    Returns
    -------
    pd.DataFrame
        Summarizing the ADF test results for each stock.
    """
    # Augmented Dickey-Fuller test
    adfuller_stats = []
    for stock in data:
        adf = adfuller(data[stock].dropna())
        adfuller_stats.append({'Ticker': stock,
                               'ADF Statistic': adf[0], 'p-value': adf[1],
                               'Stationary': 'Yes' if adf[1] < 0.05 else 'No'
                               })

    return pd.DataFrame(adfuller_stats)
