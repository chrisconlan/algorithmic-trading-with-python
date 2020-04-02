def calculate_calmar_ratio(series: pd.Series, years_past: int=3) -> float:
    """
    Return the percent max drawdown ratio over the past three years using
    CAGR as the numerator, otherwise known as the Calmar Ratio
    """

    # Filter series on past three years
    last_date = series.index[-1]
    three_years_ago = last_date - pd.Timedelta(days=years_past*365.25)
    series = series[series.index > three_years_ago]

    # Compute annualized percent max drawdown ratio
    percent_drawdown = calculate_max_drawdown(series, method='percent')
    cagr = calculate_cagr(series)
    return cagr / percent_drawdown