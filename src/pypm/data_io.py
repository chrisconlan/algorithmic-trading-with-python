import os
import pandas as pd
from pandas import DataFrame
from typing import Dict, List, Tuple

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "data",
)
EOD_DATA_DIR = os.path.join(DATA_DIR, "eod")
ALTERNATIVE_DATA_DIR = os.path.join(DATA_DIR, "alternative_data")


def load_eod_data(ticker: str, data_dir: str = EOD_DATA_DIR) -> DataFrame:
    f_path = os.path.join(data_dir, f"{ticker}.csv")
    assert os.path.isfile(f_path), f"No data available for {ticker}"
    return pd.read_csv(f_path, parse_dates=["date"], index_col="date")


def load_spy_data() -> DataFrame:
    """
    Convenience function to load S&P 500 ETF EOD data
    """
    return load_eod_data("SPY", DATA_DIR)


def _combine_columns(filepaths_by_symbol: Dict[str, str],
                     attr: str = "close") -> pd.DataFrame:

    data_frames = [
        pd.read_csv(
            filepath,
            index_col="date",
            usecols=["date", attr],
            parse_dates=["date"],
        ).rename(columns={
            "date": "date",
            attr: symbol,
        }) for symbol, filepath in list(filepaths_by_symbol.items())
    ]
    return pd.concat(data_frames, sort=True, axis=1)


def load_eod_matrix(tickers: List[str], attr: str = "close") -> pd.DataFrame:
    filepaths_by_symbol = {
        t: os.path.join(EOD_DATA_DIR, f"{t}.csv") for t in tickers
    }
    return _combine_columns(filepaths_by_symbol, attr)


def load_alternative_data_matrix(tickers: List[str]) -> pd.DataFrame:
    filepaths_by_symbol = {
        t: os.path.join(ALTERNATIVE_DATA_DIR, f"{t}.csv") for t in tickers
    }
    return _combine_columns(filepaths_by_symbol, "value")


def get_all_symbols() -> List[str]:
    return [v.strip(".csv") for v in os.listdir(EOD_DATA_DIR)]


def build_eod_closes() -> None:
    filenames = os.listdir(EOD_DATA_DIR)
    filepaths_by_symbol = {
        v.strip(".csv"): os.path.join(EOD_DATA_DIR, v) for v in filenames
    }
    result = _combine_columns(filepaths_by_symbol)
    result.to_csv(os.path.join(DATA_DIR, "eod_closes.csv"))


def concatenate_metrics(df_by_metric: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenates different dataframes that have the same columns into a
    hierarchical dataframe.

    The input df_by_metric should of the form

    {
        'metric_1': pd.DataFrame()
        'metric_2: pd.DataFrame()
    }
    where each dataframe should have the same columns, i.e. symbols.
    """

    to_concatenate = []
    tuples = []
    for key, df in list(df_by_metric.items()):
        to_concatenate.append(df)
        tuples += [(s, key) for s in df.columns.values]

    df = pd.concat(to_concatenate, sort=True, axis=1)
    df.columns = pd.MultiIndex.from_tuples(tuples, names=["symbol", "metric"])

    return df


if __name__ == "__main__":
    build_eod_closes()
