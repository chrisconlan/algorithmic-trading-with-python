def calculate_sharpe_ratio(price_series: pd.Series, 
    benchmark_rate: float=0) -> float:
    """
    Calculates the sharpe ratio given a price series. Defaults to benchmark_rate
    of zero.
    """
    cagr = calculate_cagr(price_series)
    return_series = calculate_return_series(price_series)
    volatility = calculate_annualized_volatility(return_series)
    return (cagr - benchmark_rate) / volatility