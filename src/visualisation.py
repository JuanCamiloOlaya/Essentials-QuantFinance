
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def tracking_plot(data: pd.DataFrame, tickers: list[str],
                  title: str, xlabel: str, ylabel: str):
    """Line plot for selected tickers.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame with dates as the index and tickers as columns.
    tickers : list[str]
        A list of ticker symbols to plot.
    title : str
        Title of the plot.
    xlabel : str
        x label required for the plot.
    ylabel : str
        y label required for the plot.
    """
    plt.figure(figsize=(9, 6))

    for t in tickers:
        plt.plot(data.index, data[t], label=t)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, color='k', linestyle=':')
    plt.tight_layout()
    plt.show()


def plot_histograms(data: pd.DataFrame, tickers: list[str]):
    """Creates subplots with histograms for each stock's returns.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame with dates as the index and tickers' returns as columns.
    tickers : list[str]
        A list of ticker symbols to plot.
    """
    n = len(tickers)
    rows = (n + 2) // 3
    cols = 3

    _, axes = plt.subplots(rows, cols, figsize=(9, 6))
    axes = axes.flatten()

    for i, ticker in enumerate(tickers):
        if ticker in data.columns:
            axes[i].hist(data[ticker], alpha=0.7, color='blue', edgecolor='black')
            axes[i].set_title(f'{ticker} Returns', fontsize=12)
            axes[i].set_xlabel('Return')
            axes[i].set_ylabel('Frequency')
        else:
            axes[i].axis('off')

    for j in range(len(tickers), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def plot_boxplots(data: pd.DataFrame, tickers: list[str]):
    """Creates subplots with boxplots for each stock's returns.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame with dates as the index and tickers' returns as columns.
    tickers : list[str]
        A list of ticker symbols to plot.
    """
    n = len(tickers)
    rows = (n + 2) // 3
    cols = 3

    _, axes = plt.subplots(rows, cols, figsize=(9, 6))
    axes = axes.flatten()

    for i, ticker in enumerate(tickers):
        if ticker in data.columns:
            axes[i].boxplot(data[ticker].dropna(), patch_artist=True, boxprops=dict(facecolor='lightblue'))
            axes[i].set_title(f'{ticker} Returns', fontsize=12)
            axes[i].set_ylabel('Return')
            axes[i].set_xticks([])
        else:
            axes[i].axis('off')

    # Turn off remaining unused subplots
    for j in range(len(tickers), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def ts_decomposition(data: pd.DataFrame, tickers: list[str]):
    """Perform seasonal decomposition for all tickers and plot
    the results.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame with dates as the index and tickers as columns.
    tickers : list[str]
        A list of ticker symbols to plot.
    """
    n = len(tickers)
    fig, axes = plt.subplots(n, 3, figsize=(17, 18))
    fig.tight_layout(pad=5.0)

    for i, ticker in enumerate(data.columns):
        # Seasonal decomposition
        decomposition = seasonal_decompose(data[ticker].dropna(),
                                           model='additive', period=30)

        # Plots
        axes[i, 0].plot(decomposition.trend)
        axes[i, 0].set_title(f'{ticker} - Trend')
        axes[i, 0].grid()

        axes[i, 1].plot(decomposition.seasonal)
        axes[i, 1].set_title(f'{ticker} - Seasonal')
        axes[i, 1].grid()

        axes[i, 2].plot(decomposition.resid)
        axes[i, 2].set_title(f'{ticker} - Residuals')
        axes[i, 2].grid()

    plt.suptitle('Seasonal Decomposition of Stock Prices')
    plt.show()


def acf_pacf(data: pd.DataFrame, tickers: list[str]):
    """Perform seasonal decomposition for all tickers and plot
    the results.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame with dates as the index and tickers as columns.
    tickers : list[str]
        A list of ticker symbols to plot.
    """
    n = len(tickers)
    fig, axes = plt.subplots(n, 2, figsize=(17, 18))
    fig.tight_layout(pad=5.0)

    for i, ticker in enumerate(data.columns):
        # Autocorrelation
        plot_acf(data[ticker].dropna(), lags=25, ax=axes[i, 0])
        axes[i, 0].set_title(f'{ticker} - Autocorrelation (ACF)')

        # Partial Autocorrelation
        plot_pacf(data[ticker].dropna(), lags=25, ax=axes[i, 1])
        axes[i, 1].set_title(f'{ticker} - Partial Autocorrelation (PACF)')

    plt.suptitle('ACF and PACF for Stock Returns')
    plt.show()


def moving_averages(data: pd.DataFrame, ma: pd.DataFrame, ema: pd.DataFrame):
    """Plot the original stock prices, moving average (MA), and exponential
    moving average (EMA).

    Parameters
    ----------
    data : pd.DataFrame
        The original stock prices.
    ma : pd.DataFrame
        The calculated moving averages for each stock.
    ema : pd.DataFrame
        The calculated exponential moving averages for each stock.
    """
    n = data.shape[1]
    _, axes = plt.subplots(n, 1, figsize=(9, 23))

    if n == 1:
        axes = [axes]

    for i, ticker in enumerate(data.columns):
        axes[i].plot(data.index, data[ticker], label=f'{ticker} - Prices', color='blue', linewidth=0.7)
        axes[i].plot(ma.index, ma[ticker], label=f'{ticker} - MA', color='orange', linestyle='--')
        axes[i].plot(ema.index, ema[ticker], label=f'{ticker} - EMA', color='green', linestyle='-.')
        axes[i].set_title(f'{ticker} - Moving Averages')
        axes[i].legend()
        axes[i].grid()

    plt.tight_layout()
    plt.show()

