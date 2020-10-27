import pandas as pd

from pypm import metrics, signals, data_io, simulation, optimization
from pypm.optimization import GridSearchOptimizer

from typing import List, Dict, Tuple, Callable

# Dict[str, float]
Performance = simulation.PortfolioHistory.PerformancePayload


def bind_simulator(**sim_kwargs) -> Callable:
    """
    Create a function with all static simulation data bound to it, where the
    arguments are simulation parameters
    """

    symbols: List[str] = data_io.get_all_symbols()
    prices: pd.DataFrame = data_io.load_eod_matrix(symbols)

    _bollinger: Callable = signals.create_bollinger_band_signal
    _sharpe: Callable = metrics.calculate_rolling_sharpe_ratio

    def _simulate(bollinger_n: int, sharpe_n: int) -> Performance:

        signal = prices.apply(_bollinger, args=(bollinger_n,), axis=0)
        preference = prices.apply(_sharpe, args=(sharpe_n,), axis=0)

        simulator = simulation.SimpleSimulator(**sim_kwargs)
        simulator.simulate(prices, signal, preference)

        return simulator.portfolio_history.get_performance_metric_data()

    return _simulate


if __name__ == "__main__":

    simulate = bind_simulator(initial_cash=10000, max_active_positions=5)

    optimizer = GridSearchOptimizer(simulate)
    optimizer.optimize(
        bollinger_n=list(range(10, 110, 10)),
        sharpe_n=list(range(10, 110, 10)),
    )

    print((optimizer.get_best("excess_cagr")))
    optimizer.plot("excess_cagr")
    optimizer.plot("bollinger_n", "excess_cagr")
    optimizer.plot("bollinger_n", "sharpe_n", "excess_cagr")

# Returns ...
#     bollinger_n  sharpe_n  excess_cagr
# 17           20        80     0.092841
# 16           20        70     0.062477
# 98          100        90     0.055047
# 19           20       100     0.050255
# 1            10        20     0.043642
# ..          ...       ...          ...
# 89           90       100    -0.054080
# 69           70       100    -0.054404
# 63           70        40    -0.061105
# 61           70        20    -0.063276
# 50           60        10    -0.065433
