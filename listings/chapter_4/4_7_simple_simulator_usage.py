# pypm/simulate_portfolio.py
from pypm import metrics, signals, data_io, simulation
import pandas as pd


def simulate_portfolio():

    bollinger_n = 20
    sharpe_n = 20

    # Load in data
    symbols: List[str] = data_io.get_all_symbols()
    prices: pd.DataFrame = data_io.load_eod_matrix(symbols)

    # Use the bollinger band outer band crossorver as a signal
    _bollinger = signals.create_bollinger_band_signal
    signal = prices.apply(_bollinger, args=(bollinger_n,), axis=0)

    # Use a rolling sharpe ratio approximation as a preference matrix
    _sharpe = metrics.calculate_rolling_sharpe_ratio
    preference = prices.apply(_sharpe, args=(sharpe_n,), axis=0)

    # Run the simulator
    simulator = simulation.SimpleSimulator(
        initial_cash=10000,
        max_active_positions=5,
        percent_slippage=0.0005,
        trade_fee=1,
    )
    simulator.simulate(prices, signal, preference)

    # Print results
    simulator.portfolio_history.print_position_summaries()
    simulator.print_initial_parameters()
    simulator.portfolio_history.print_summary()
    simulator.portfolio_history.plot()


if __name__ == "__main__":
    simulate_portfolio()

# Returns ...
# Initial Cash: $10000
# Maximum Number of Assets: 5
#
# Equity: $39758.61
# Percent Return: 297.59%
# S&P 500 Return: 184.00%
#
# Number of trades: 1835
# Average active trades: 4.83
#
# CAGR: 14.82%
# S&P 500 CAGR: 11.02%
# Excess CAGR: 3.80%
#
# Annualized Volatility: 17.93%
# Sharpe Ratio: 0.83
# Jensen's Alpha: 0.000147
#
# Dollar Max Drawdown: $10594.83
# Percent Max Drawdown: 30.03%
# Log Max Drawdown Ratio: 1.02
