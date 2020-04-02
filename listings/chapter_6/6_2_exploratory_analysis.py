from pypm import data_io, metrics
import numpy as np
import pandas as pd
from typing import List

# Load in everything
symbols: List[str] = data_io.get_all_symbols()
eod_data: pd.DataFrame = data_io.load_eod_matrix(symbols)
alt_data: pd.DataFrame = data_io.load_alternative_data_matrix(symbols)
eod_data = eod_data[eod_data.index >= alt_data.index.min()]

_calc_returns = metrics.calculate_log_return_series
_corr_by_symbol = dict()

for symbol in symbols:

    alt_series = alt_data[symbol].dropna()
    price_series = eod_data[symbol]

    if alt_series.empty:
        continue

    # Calculate returns, ensuring each series has the same index
    price_return_series = _calc_returns(price_series.loc[alt_series.index])
    alt_return_series = _calc_returns(alt_series)

    # Remove the NA at the front
    price_return_series = price_return_series.iloc[1:]
    alt_return_series = alt_return_series.iloc[1:]

    # Calculate the correllation
    _corr = np.corrcoef(price_return_series, alt_return_series)

    # This element of the correlation matrix is the number we want
    _corr_by_symbol[symbol] = _corr[1,0]

# Describe results
results = pd.Series(_corr_by_symbol)
print(pd.DataFrame(results.describe()).T)
# Returns ...
#  count      mean       std       min       25%       50%     75%       max
#   97.0 -0.002539  0.032456 -0.065556 -0.024983 -0.003735  0.0174  0.099085