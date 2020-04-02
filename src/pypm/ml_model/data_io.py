import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Tuple, List

from pypm import data_io

def load_data() -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
	"""
	Load the data as is will be used in the alternative data model
	"""
	symbols: List[str] = data_io.get_all_symbols()
	alt_data = data_io.load_alternative_data_matrix(symbols)
	eod_data = data_io.load_eod_matrix(symbols)
	eod_data = eod_data[eod_data.index >= alt_data.index.min()]

	return symbols, eod_data, alt_data
