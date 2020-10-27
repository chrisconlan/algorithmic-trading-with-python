import numpy as np
import pandas as pd
from pypm import filters


def calculate_events_for_revenue_series(
        series: pd.Series,
        filter_threshold: float,
        lookback: int = 365) -> pd.DatetimeIndex:
    """
    Calculate the symmetric cusum filter to generate events on YoY changes in
    the log revenue series
    """
    series = np.log(series)
    series = filters.calculate_non_uniform_lagged_change(series, lookback)
    return filters.calculate_cusum_events(series, filter_threshold)


def calculate_events(revenue_series: pd.Series):
    return calculate_events_for_revenue_series(
        revenue_series,
        filter_threshold=5,
        lookback=365,
    )
