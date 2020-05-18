import pandas as pd
import numpy as np

from pypm.ml_model.events import calculate_events
from pypm.ml_model.features import calculate_features

from typing import List

def calculate_signals(classifier, symbols: List[str], eod_data: pd.DataFrame,
	alt_data: pd.DataFrame) -> pd.DataFrame:
	"""
	Calculate signal dataframes for use in the simulator
	"""

	# For storing the signals
	signal_by_symbol = dict()

	# Build events and features for each symbol
	for symbol in symbols:

		# Get revenue and price series
		revenue_series = alt_data[symbol].dropna()
		price_series = eod_data[symbol].dropna()

		# Build output template
		signal_series = pd.Series(0, index=price_series.index)

		# Get events and features
		event_index = calculate_events(revenue_series)
		features_df = calculate_features(price_series, revenue_series)

		features_on_events = features_df.loc[event_index]
		features_on_events.dropna(inplace=True)
		event_index = features_on_events.index

		if features_on_events.empty:
			predictions = pd.Series()
		else:
			_predictions = classifier.predict(features_on_events)
			predictions = pd.Series(_predictions, index=event_index)
		
		# Add into output template
		signal_series = signal_series.add(predictions, fill_value=0)

		signal_by_symbol[symbol] = signal_series

	signal = pd.DataFrame(signal_by_symbol)
	signal.sort_index(inplace=True)
	signal.dropna(inplace=True)

	return signal

