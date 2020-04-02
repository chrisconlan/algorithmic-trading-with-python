def calculate_cagr(series: pd.Series) -> float:
    """
    Calculate compounded annual growth rate
    """
    value_factor = series.iloc[-1] / series.iloc[0]
    year_past = get_years_past(series)
    return (value_factor ** (1 / year_past)) - 1