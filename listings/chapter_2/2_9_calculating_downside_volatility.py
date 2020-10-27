def calculate_annualized_downside_deviation(return_series: pd.Series,
                                            benchmark_rate: float = 0) -> float:
    """
    Calculates the downside deviation for use in the sortino ratio.

    Benchmark rate is assumed to be annualized. It will be adjusted according
    to the number of periods per year seen in the data.
    """

    # For both de-annualizing the benchmark rate and annualizing result
    years_past = get_years_past(return_series)
    entries_per_year = return_series.shape[0] / years_past

    adjusted_benchmark_rate = ((1 + benchmark_rate)**(1 / entries_per_year)) - 1

    downside_series = adjusted_benchmark_rate - return_series
    downside_sum_of_squares = (downside_series[downside_series > 0]**2).sum()
    denominator = return_series.shape[0] - 1
    downside_deviation = np.sqrt(downside_sum_of_squares / denominator)

    return downside_deviation * np.sqrt(entries_per_year)


def calculate_sortino_ratio(price_series: pd.Series,
                            benchmark_rate: float = 0) -> float:
    """
    Calculates the sortino ratio.
    """
    cagr = calculate_cagr(price_series)
    return_series = calculate_return_series(price_series)
    downside_deviation = calculate_annualized_downside_deviation(return_series)
    return (cagr - benchmark_rate) / downside_deviation
