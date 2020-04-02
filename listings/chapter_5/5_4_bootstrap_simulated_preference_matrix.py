import pandas as pd
import numpy as np

from pypm import metrics, signals, data_io, simulation, optimization
from pypm.optimization import GridSearchOptimizer

from typing import List, Dict, Tuple, Callable

Performance = simulation.PortfolioHistory.PerformancePayload # Dict[str, float]

def bind_simulator(**sim_kwargs) -> Callable:
    """
    Create a simulator that uses white noise for the preference matrix
    """
    symbols: List[str] = data_io.get_all_symbols()
    prices: pd.DataFrame = data_io.load_eod_matrix(symbols)

    _bollinger: Callable = signals.create_bollinger_band_signal
    bollinger_n = 20

    returns = metrics.calculate_return_series(prices)
    sharpe_n = 20

    def bootstrap_rolling_sharpe_ratio(return_series: pd.Series) -> pd.Series:
        _series = return_series.iloc[1:]
        _series = _series.sample(n=return_series.shape[0], replace=True)
        _series.iloc[:1] = [np.nan]
        _series = pd.Series(_series.values, index=return_series.index)
        _windowed_series = _series.rolling(sharpe_n)
        return _windowed_series.mean() / _windowed_series.std()

    _sharpe: Callable = bootstrap_rolling_sharpe_ratio

    def _simulate(bootstrap_test_id: int) -> Performance:
        
        signal = prices.apply(_bollinger, args=(bollinger_n,), axis=0)
        preference = returns.apply(_sharpe, axis=0)

        simulator = simulation.SimpleSimulator(**sim_kwargs)
        simulator.simulate(prices, signal, preference)

        return simulator.portfolio_history.get_performance_metric_data()

    return _simulate

if __name__ == '__main__':

    simulate = bind_simulator(initial_cash=10000, max_active_positions=5)

    optimizer = GridSearchOptimizer(simulate)
    optimizer.optimize(bootstrap_test_id=range(1000))

    print(optimizer.get_best('excess_cagr'))
    optimizer.print_summary()
    optimizer.plot('excess_cagr')