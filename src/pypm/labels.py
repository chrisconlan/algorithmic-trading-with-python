import numpy as np
import pandas as pd
from typing import Tuple

def compute_triple_barrier_labels(
    price_series: pd.Series, 
    event_index: pd.Series, 
    time_delta_days: int, 
    upper_delta: float=None, 
    lower_delta: float=None, 
    vol_span: int=20, 
    upper_z: float=None,
    lower_z: float=None,
    upper_label: int=1, 
    lower_label: int=-1) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate event labels according to the triple-barrier method. 

    Return a series with both the original events and the labels. Labels 1, 0, 
    and -1 correspond to upper barrier breach, vertical barrier breach, and 
    lower barrier breach, respectively. 

    Also return series where the index is the start date of the label and the 
    values are the end dates of the label.
    """

    timedelta = pd.Timedelta(days=time_delta_days)
    series = pd.Series(np.log(price_series.values), index=price_series.index)

    # A list with elements of {-1, 0, 1} indicating the outcome of the events
    labels = list()
    label_dates = list()

    if upper_z or lower_z:
        volatility = series.ewm(span=vol_span).std()
        volatility *= np.sqrt(time_delta_days / vol_span)

    for event_date in event_index:
        date_barrier = event_date + timedelta

        start_price = series.loc[event_date]
        log_returns = series.loc[event_date:date_barrier] - start_price

        # First element of tuple is 1 or -1 indicating upper or lower barrier
        # Second element of tuple is first date when barrier was crossed
        candidates: List[Tuple[int, pd.Timestamp]] = list()

        # Add the first upper or lower date to candidates
        if upper_delta:
            _date = log_returns[log_returns > upper_delta].first_valid_index()
            if _date:
                candidates.append((upper_label, _date))
    
        if lower_delta:
            _date = log_returns[log_returns < lower_delta].first_valid_index()
            if _date:
                candidates.append((lower_label, _date))

        # Add the first upper_z and lower_z to candidates
        if upper_z:
            upper_barrier = upper_z * volatility[event_date]
            _date = log_returns[log_returns > upper_barrier].first_valid_index()
            if _date:
                candidates.append((upper_label, _date))

        if lower_z:
            lower_barrier = lower_z * volatility[event_date]
            _date = log_returns[log_returns < lower_barrier].first_valid_index()
            if _date:
                candidates.append((lower_label, _date))

        if candidates:
            # If any candidates, return label for first date
            label, label_date = min(candidates, key=lambda x: x[1])
        else:
            # If there were no candidates, time barrier was touched
            label, label_date = 0, date_barrier

        labels.append(label)
        label_dates.append(label_date)

    label_series = pd.Series(labels, index=event_index)
    event_spans = pd.Series(label_dates, index=event_index)

    return label_series, event_spans





