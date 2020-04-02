def calculate_return_series(series: pd.Series) -> pd.Series:
    """
    Calculates the return series of a time series.
    The first value will always be NaN.
    Output series retains the index of the input series.
    """
    shifted_series = series.shift(1, axis=0)
    return series / shifted_series - 1