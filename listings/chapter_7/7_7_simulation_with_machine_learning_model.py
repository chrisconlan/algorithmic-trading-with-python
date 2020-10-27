import pandas as pd
import numpy as np

import os
from joblib import load

from pypm.ml_model.data_io import load_data
from pypm.ml_model.signals import calculate_signals

from pypm import metrics, simulation

SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def simulate_portfolio():

    # All the data we have to work with
    symbols, eod_data, alt_data = load_data()

    # Load classifier from file
    classifier = load(os.path.join(SRC_DIR, "ml_model.joblib"))

    # Generate signals from classifier
    print("Calculating signals ...")
    signal = calculate_signals(classifier, symbols, eod_data, alt_data)

    # Get rid of eod_data before valid signals
    first_signal_date = signal.first_valid_index()
    eod_data = eod_data[eod_data.index > first_signal_date]

    # Set the preference to increase by row, so new trades are preferred
    print("Calculating preference matrix ...")
    preference = pd.DataFrame(
        np.random.random(eod_data.shape),
        columns=eod_data.columns,
        index=eod_data.index,
    )

    # Run the simulator
    simulator = simulation.SimpleSimulator(
        initial_cash=10000,
        max_active_positions=10,
        percent_slippage=0.0005,
        trade_fee=1,
    )
    simulator.simulate(eod_data, signal, preference)

    # Print results
    simulator.portfolio_history.print_position_summaries()
    simulator.print_initial_parameters()
    simulator.portfolio_history.print_summary()
    simulator.portfolio_history.plot()
    simulator.portfolio_history.plot_benchmark_comparison()


if __name__ == "__main__":
    simulate_portfolio()

# Returns ...
# Initial Cash: $10000
# Maximum Number of Assets: 10
#
# Equity: $45455.68
# Percent Return: 354.56%
# S&P 500 Return: 33.80%
#
# Number of trades: 291
# Average active trades: 9.89
#
# CAGR: 83.75%
# S&P 500 CAGR: 12.43%
# Excess CAGR: 71.32%
#
# Annualized Volatility: 14.44%
# Sharpe Ratio: 5.80
# Jensen's Alpha: 0.002018
#
# Dollar Max Drawdown: $1892.59
# Percent Max Drawdown: 8.60%
# Log Max Drawdown Ratio: 1.42
#
