import numpy as np
import pandas as pd
from scipy.stats import hmean

def calculate_uniqueness(event_spans: pd.Series, 
    price_index: pd.Series) -> pd.Series:
    """
    event_spans is a series with an index of start dates and values of end dates
    of a label.

    price_index is an index of underlying dates for the event

    Returns a series of uniqueness values that can be used as weights, indexed 
    as the event start dates. Weights may need to be standardized again before 
    training.
    """

    # Create a binary data frame 
    # value is 1 during event span and 0 otherwise
    columns = range(event_spans.shape[0])
    df = pd.DataFrame(0, index=price_index, columns=columns)

    for i, (event_start, event_end) in enumerate(event_spans.items()):
        df[i].loc[event_start:event_end] += 1

    # Compute concurrency over event span then calculate uniqueness
    avg_uniquenesses = list()
    for i, (event_start, event_end) in enumerate(event_spans.items()):
        concurrency: pd.Series = df.loc[event_start:event_end].sum(axis=1)
        avg_uniqueness = 1 / hmean(concurrency)
        avg_uniquenesses.append(avg_uniqueness)

    return pd.Series(avg_uniquenesses, index=event_spans.index)
