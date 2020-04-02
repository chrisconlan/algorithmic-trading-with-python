import numpy as np
import pandas as pd

def calculate_non_uniform_lagged_change(series: pd.Series, n_days: int):
    """
    Use pd.Series.searchsorted to measure the lagged change in a non-uniformly 
    spaced time series over n_days of calendar time. 
    """

    # Get mapping from now to n_days ago at every point
    _timedelta: pd.Timedelta = pd.Timedelta(days=n_days)
    _idx: pd.Series = series.index.searchsorted(series.index - _timedelta)
    _idx = _idx[_idx > 0]

    # Get the last len(series) - n_days values
    _series = series.iloc[-_idx.shape[0]:]

    # Build a padding of NA values
    _pad_length = series.shape[0] - _idx.shape[0]
    _na_pad = pd.Series(None, index=series.index[:_pad_length])

    # Get the corresonding lagged values
    _lagged_series = series.iloc[_idx]

    # Measure the difference
    _diff = pd.Series(_series.values-_lagged_series.values, index=_series.index)

    return pd.concat([_na_pad, _diff])


def calculate_cusum_events(series: pd.Series, 
    filter_threshold: float) -> pd.DatetimeIndex:
    """
    Calculate symmetric cusum filter and corresponding events
    """

    event_dates = list()
    s_up = 0
    s_down = 0

    for date, price in series.items():
        s_up = max(0, s_up + price)
        s_down = min(0, s_down + price)

        if s_up > filter_threshold:
            s_up = 0
            event_dates.append(date)

        elif s_down < -filter_threshold:
            s_down = 0
            event_dates.append(date)

    return pd.DatetimeIndex(event_dates)


