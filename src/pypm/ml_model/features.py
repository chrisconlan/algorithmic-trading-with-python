import numpy as np
import pandas as pd

from pypm import indicators, filters, metrics

_calc_delta = filters.calculate_non_uniform_lagged_change
_calc_ma = indicators.calculate_simple_moving_average
_calc_log_return = metrics.calculate_log_return_series


def _calc_rolling_vol(series, n):
    return series.rolling(n).std() * np.sqrt(252 / n)


def calculate_features(price_series, revenue_series) -> pd.DataFrame:
    """
    Calculate any and all potentially useful features. Return as a dataframe.
    """

    log_revenue = np.log(revenue_series)
    log_prices = np.log(price_series)

    log_revenue_ma = _calc_ma(log_revenue, 10)
    log_prices_ma = _calc_ma(log_prices, 10)

    log_returns = _calc_log_return(price_series)

    features_by_name = dict()

    for i in [7, 30, 90, 180, 360]:

        rev_feature = _calc_delta(log_revenue_ma, i)
        price_feature = _calc_delta(log_prices_ma, i)
        vol_feature = _calc_rolling_vol(log_returns, i)

        features_by_name.update({
            f"{i}_day_revenue_delta": rev_feature,
            f"{i}_day_return": price_feature,
            f"{i}_day_vol": vol_feature,
        })

    features_df = pd.DataFrame(features_by_name)
    return features_df
