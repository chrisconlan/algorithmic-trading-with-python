# See fit_alternative_data_model.py
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

        # Convert labels and events to a data frame
        labels_df = pd.DataFrame(event_labels)
        labels_df.columns = ['y']

        # Converts weights to a data frame
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

# Returns ...
#             7_day_revenue_delta  7_day_return  7_day_vol  ...
# 2016-06-07            -0.000721      0.019520   0.096002  ...
# 2016-06-08             0.029827      0.025005   0.113246  ...
# 2016-06-08            -0.046427      0.013868   0.051878  ...
# 2016-06-09             0.001558      0.032410   0.064574  ...
# 2016-06-10             0.004933      0.011751   0.045105  ...
# ...                         ...           ...        ...  ...
# 2019-09-30            -0.031956     -0.008562   0.072845  ...
# 2019-10-01            -0.074244     -0.018469   0.053665  ...
# 2019-10-01             0.009513     -0.015659   0.094087  ...
# 2019-10-02             0.012819     -0.008300   0.062938  ...
# 2019-10-02             0.003023      0.015749   0.043320  ...
# 
# [1563 rows x 17 columns]
# Fitting 20 models 10 at a time ...
# 
# ...
# ...
# ...
# 
# Feature importances
# 30_day_return            0.099
# 7_day_return             0.097
# 30_day_vol               0.073
# 90_day_return            0.068
# 360_day_vol              0.066
# 360_day_revenue_delta    0.064
# 360_day_return           0.063
# 180_day_return           0.063
# 180_day_revenue_delta    0.060
# 90_day_vol               0.060
# 180_day_vol              0.060
# 7_day_vol                0.059
# 7_day_revenue_delta      0.057
# 90_day_revenue_delta     0.057
# 30_day_revenue_delta     0.055
# 
# Cross validation scores
# ...
# 
# Baseline accuracy 42.2%
# OOS accuracy 52.4% +/- 5.3%
# Improvement 4.9 to 15.6%
# 