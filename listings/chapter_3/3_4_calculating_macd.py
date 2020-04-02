def calculate_macd_oscillator(series: pd.Series,
    n1: int=5, n2: int=34) -> pd.Series:
    """
    Calculate the moving average convergence divergence oscillator, given a 
    short moving average of length n1 and a long moving average of length n2
    """
    assert n1 < n2, f'n1 must be less than n2'
    return calculate_simple_moving_average(series, n1) - \
        calculate_simple_moving_average(series, n2)