import os
import pandas as pd
import numpy as np
from typing import Dict

from joblib import dump

from pypm.ml_model.data_io import load_data
from pypm.ml_model.events import calculate_events
from pypm.ml_model.labels import calculate_labels
from pypm.ml_model.features import calculate_features
from pypm.ml_model.model import calculate_model
from pypm.ml_model.weights import calculate_weights

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':

    # All the data we have to work with
    symbols, eod_data, alt_data = load_data()

    # The ML dataframe for each symbol, to be combined later
    df_by_symbol: Dict[str, pd.DataFrame] = dict()

    # Build ML dataframe for each symbol
    for symbol in symbols:

        # Get revenue and price series
        revenue_series = alt_data[symbol].dropna()
        price_series = eod_data[symbol].dropna()
        price_index = price_series.index

        # Get events, labels, weights, and features
        event_index = calculate_events(revenue_series)
        event_labels, event_spans = calculate_labels(price_series, event_index)
        weights = calculate_weights(event_spans, price_index)
        features_df = calculate_features(price_series, revenue_series)

        # Subset features by event dates
        features_on_events = features_df.loc[event_index]

        # Convert labels and events to a dataframe
        labels_df = pd.DataFrame(event_labels)
        labels_df.columns = ['y']

        # Converts weights to a dataframe
        weights_df = pd.DataFrame(weights)
        weights_df.columns = ['weights']

        # Concatenate features to labels
        df = pd.concat([features_on_events, weights_df, labels_df], axis=1)
        df_by_symbol[symbol] = df

    # Create final ML dataframe
    df = pd.concat(df_by_symbol.values(), axis=0)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    print(df)

    # Fit the model
    classifier = calculate_model(df)

    # Save the model
    dump(classifier, os.path.join(SRC_DIR, 'ml_model.joblib'))


