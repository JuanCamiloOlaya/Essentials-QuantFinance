
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta


def stock_data(tickers: list[str], days=200) -> pd.DataFrame:
    """Download historical data for the specified tickers for a given period.

    Parameters
    ----------
    tickers : list[str]
        List of stock symbols to download.
    days : int, optional
        Time period for data, by default 200 days.

    Returns
    -------
    pd.DataFrame
        DataFrame with adjusted close and volume of each stock in the
        specified period.
    """
    end = datetime.today()
    start = end - timedelta(days=days)

    try:
        data = yf.download(tickers, start=start, end=end)
        data.index.name = None
        data.columns.name = None
        return data
    except Exception as E:
        print(f"Error al descargar datos: {E}")
        return None

