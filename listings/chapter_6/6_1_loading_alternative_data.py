from pypm import data_io
import numpy as np
import pandas as pd
from typing import List

# Load in everything
symbols: List[str] = data_io.get_all_symbols()
eod_data: pd.DataFrame = data_io.load_eod_matrix(symbols)
alt_data: pd.DataFrame = data_io.load_alternative_data_matrix(symbols)

# Our eod_data goes back 10 years, but our alt_data goes back 5 years
eod_data = eod_data[eod_data.index >= alt_data.index.min()]
assert np.all(eod_data.index == alt_data.index)
assert np.all(eod_data.columns == alt_data.columns)