import pandas as pd
from pypm.data_io import load_eod_data


def calculate_simple_moving_average(series: pd.Series,
                                    n: int = 20) -> pd.Series:
    """Calculates the simple moving average"""
    return series.rolling(n).mean()


def calculate_simple_moving_sample_stdev(series: pd.Series,
                                         n: int = 20) -> pd.Series:
    """Calculates the simple moving average"""
    return series.rolling(n).std()


def calculate_macd_oscillator(series: pd.Series,
                              n1: int = 5,
                              n2: int = 34) -> pd.Series:
    """
    Calculate the moving average convergence divergence oscillator, given a
    short moving average of length n1 and a long moving average of length n2
    """
    assert n1 < n2, f"n1 must be less than n2"
    return calculate_simple_moving_average(
        series, n1) - calculate_simple_moving_average(series, n2)


def calculate_bollinger_bands(series: pd.Series, n: int = 20) -> pd.DataFrame:
    """
    Calculates the Bollinger Bands and returns them as a dataframe
    """

    sma = calculate_simple_moving_average(series, n)
    stdev = calculate_simple_moving_sample_stdev(series, n)

    return pd.DataFrame({
        "middle": sma,
        "upper": sma + 2 * stdev,
        "lower": sma - 2 * stdev
    })


def calculate_money_flow_volume_series(df: pd.DataFrame) -> pd.Series:
    """
    Calculates money flow series
    """
    mfv = (df["volume"] * (2 * df["close"] - df["high"] - df["low"]) /
           (df["high"] - df["low"]))
    return mfv


def calculate_money_flow_volume(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """
    Calculates money flow volume, or q_t in our formula
    """
    return calculate_money_flow_volume_series(df).rolling(n).sum()


def calculate_chaikin_money_flow(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """
    Calculates the Chaikin money flow
    """
    return calculate_money_flow_volume(df, n) / df["volume"].rolling(n).sum()


if __name__ == "__main__":
    data = load_eod_data("AWU")
    closes = data["close"]
    sma = calculate_simple_moving_average(closes, 10)
    macd = calculate_macd_oscillator(closes, 5, 50)

    bollinger_bands = calculate_bollinger_bands(closes, 100)
    bollinger_bands = bollinger_bands.assign(closes=closes)
    bollinger_bands.plot()

    cmf = calculate_chaikin_money_flow(data)
    # cmf.plot()

    import matplotlib.pyplot as plt

    plt.show()
