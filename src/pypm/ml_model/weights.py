import numpy as np
import pandas as pd

from pypm.weights import calculate_uniqueness

def calculate_weights(event_spans: pd.Series, 
    price_index: pd.Series) -> pd.Series:
    return calculate_uniqueness(event_spans, price_index)
