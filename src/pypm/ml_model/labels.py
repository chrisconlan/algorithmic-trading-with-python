import numpy as np
import pandas as pd
from typing import Tuple

from pypm import labels


def calculate_labels(price_series, event_index) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate labels based on the triple barrier method. Return a series of
    event labels index by event start date, and return a series of event end
    dates indexed by event start date.
    """

    # Remove event that don't have a proper chance to materialize
    time_delta_days = 90
    max_date = price_series.index.max()
    cutoff = max_date - pd.Timedelta(days=time_delta_days)
    event_index = event_index[event_index <= cutoff]

    # Use triple barrier method
    event_labels, event_spans = labels.compute_triple_barrier_labels(
        price_series,
        event_index,
        time_delta_days=time_delta_days,
        # upper_delta=0.10,
        # lower_delta=-0.10,
        upper_z=1.8,
        lower_z=-1.8,
        lower_label=-1,
    )

    return event_labels, event_spans
