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

    # Bollinger n is constant throughout
    bollinger_n = 20

    def _simulate(white_noise_test_id: int) -> Performance:
        
        signal = prices.apply(_bollinger, args=(bollinger_n,), axis=0)

        # Build a pile of noise in the same shape as the price data
        _noise = np.random.normal(loc=0, scale=1, size=prices.shape)
        _cols = prices.columns
        _index = prices.index
        preference = pd.DataFrame(_noise, columns=_cols, index=_index)

        simulator = simulation.SimpleSimulator(**sim_kwargs)
        simulator.simulate(prices, signal, preference)

        return simulator.portfolio_history.get_performance_metric_data()

    return _simulate

if __name__ == '__main__':

    simulate = bind_simulator(initial_cash=10000, max_active_positions=5)

    optimizer = GridSearchOptimizer(simulate)
    optimizer.optimize(white_noise_test_id=range(1000))

    print(optimizer.get_best('excess_cagr'))
    optimizer.print_summary()
    optimizer.plot('excess_cagr')
