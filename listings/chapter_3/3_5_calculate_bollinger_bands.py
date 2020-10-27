def calculate_bollinger_bands(series: pd.Series, n: int = 20) -> pd.DataFrame:
    """
    Calculates the bollinger bands and returns them as a dataframe
    """

    sma = calculate_simple_moving_average(series, n)
    stdev = calculate_simple_moving_sample_stdev(series, n)

    return pd.DataFrame({
        "middle": sma,
        "upper": sma + 2 * stdev,
        "lower": sma - 2 * stdev
    })
