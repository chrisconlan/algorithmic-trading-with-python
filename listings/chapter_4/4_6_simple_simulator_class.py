from typing import Tuple, List, Dict, Callable, NewType, Any, Iterable

import pandas as pd
import matplotlib.pyplot as plt

from pypm import metrics, signals, data_io
from pypm.portfolio import PortfolioHistory, Position, Symbol, Dollars

from collections import OrderedDict, defaultdict


class SimpleSimulator(object):
    """
    A simple trading simulator to work with the PortfolioHistory class
    """

    def __init__(
        self,
        initial_cash: float = 10000,
        max_active_positions: int = 5,
        percent_slippage: float = 0.0005,
        trade_fee: float = 1,
    ):

        # Set simulation parameters

        # Initial cash in porfolio
        # self.cash will fluctuate
        self.initial_cash = self.cash = initial_cash

        # Maximum number of different assets that can be help simultaneously
        self.max_active_positions: int = max_active_positions

        # The percentage difference between closing price and fill price for the
        # position, to simulate adverse effects of market orders
        self.percent_slippage = percent_slippage

        # The fixed fee in order to open a position in dollar terms
        self.trade_fee = trade_fee

        # Keep track of live trades
        self.active_positions_by_symbol: Dict[Symbol, Position] = OrderedDict()

        # Keep track of portfolio history like cash, equity, and positions
        self.portfolio_history = PortfolioHistory()

    @property
    def active_positions_count(self):
        return len(self.active_positions_by_symbol)

    @property
    def free_position_slots(self):
        return self.max_active_positions - self.active_positions_count

    @property
    def active_symbols(self) -> List[Symbol]:
        return list(self.active_positions_by_symbol.keys())

    def print_initial_parameters(self):
        s = (f"Initial Cash: ${self.initial_cash} \n"
             f"Maximum Number of Assets: {self.max_active_positions}\n")
        print(s)
        return s

    @staticmethod
    def make_tuple_lookup(columns) -> Callable[[str, str], int]:
        """
        Map a multi-index dataframe to an itertuples-like object.

        The index of the dateframe is always the zero-th element.
        """

        # col is a hierarchical column index represented by a tuple of strings
        tuple_lookup: Dict[Tuple[str, str],
                           int] = {col: i + 1 for i, col in enumerate(columns)}

        return lambda symbol, metric: tuple_lookup[(symbol, metric)]

    @staticmethod
    def make_all_valid_lookup(_idx: Callable):
        """
        Return a function that checks for valid data, given a lookup function
        """
        return lambda row, symbol: (not pd.isna(
            row[_idx(symbol, "pref")]) and not pd.isna(row[_idx(
                symbol, "signal")]) and not pd.isna(row[_idx(symbol, "price")]))

    def buy_to_open(self, symbol, date, price):
        """
        Keep track of new position, make sure it isn't an existing position.
        Verify you have cash.
        """

        # Figure out how much we are willing to spend
        cash_to_spend = self.cash / self.free_position_slots
        cash_to_spend -= self.trade_fee

        # Calculate buy_price and number of shares. Fractional shares allowed.
        purchase_price = (1 + self.percent_slippage) * price
        shares = cash_to_spend / purchase_price

        # Spend the cash
        self.cash -= cash_to_spend + self.trade_fee
        assert self.cash >= 0, "Spent cash you do not have."
        self.portfolio_history.record_cash(date, self.cash)

        # Record the position
        positions_by_symbol = self.active_positions_by_symbol
        assert symbol in positions_by_symbol, "Symbol already not in portfolio."
        position = Position(symbol, date, purchase_price, shares)
        positions_by_symbol[symbol] = position

    def sell_to_close(self, symbol, date, price):
        """
        Keep track of exit price, recover cash, close position, and record it in
        portfolio history.

        Will raise a KeyError if symbol isn't an active position
        """

        # Exit the position
        positions_by_symbol = self.active_positions_by_symbol
        position = positions_by_symbol[symbol]
        position.exit(date, price)

        # Receive the cash
        sale_value = position.last_value * (1 - self.percent_slippage)
        self.cash += sale_value
        self.portfolio_history.record_cash(date, self.cash)

        # Record in portfolio history
        self.portfolio_history.add_to_history(position)
        del positions_by_symbol[symbol]

    @staticmethod
    def _assert_equal_columns(*args: Iterable[pd.DataFrame]):
        column_names = set(args[0].columns.values)
        for arg in args[1:]:
            assert (set(arg.columns.values) == column_names
                   ), "Found unequal column names in input data frames."

    def simulate(self, price: pd.DataFrame, signal: pd.DataFrame,
                 preference: pd.DataFrame):
        """
        Runs the simulation.

        price, signal, and preference are data frames with the column names
        represented by the same set of stock symbols.
        """

        # Create a hierarchical data frame to loop through
        self._assert_equal_columns(price, signal, preference)
        df = data_io.concatenate_metrics({
            "price": price,
            "signal": signal,
            "pref": preference,
        })

        # Get list of symbols
        all_symbols = list(set(price.columns.values))

        # Get lookup functions
        _idx = self.make_tuple_lookup(df.columns)
        _all_valid = self.make_all_valid_lookup(_idx)

        # Store some variables
        active_positions_by_symbol = self.active_positions_by_symbol
        max_active_positions = self.max_active_positions

        # Iterating over all dates.
        # itertuples() is significantly faster than iterrows(), it however comes
        # at the cost of being able index easily. In order to get around this
        # we use an tuple lookup function: "_idx"
        for row in df.itertuples():

            # date index is always first element of tuple row
            date = row[0]

            # Get symbols with valid and tradable data
            symbols: List[str] = [s for s in all_symbols if _all_valid(row, s)]

            # Iterate over active positions and sell stocks with a sell signal.
            _active = self.active_symbols
            to_exit = [s for s in _active if row[_idx(s, "signal")] == -1]
            for s in to_exit:
                sell_price = row[_idx(s, "price")]
                self.sell_to_close(s, date, sell_price)

            # Get up to max_active_positions symbols with a buy signal in
            # decreasing order of preference
            to_buy = [
                s for s in symbols if row[_idx(s, "signal")] == 1 and
                s not in active_positions_by_symbol
            ]
            to_buy.sort(key=lambda s: row[_idx(s, "pref")], reverse=True)
            to_buy = to_buy[:max_active_positions]

            for s in to_buy:
                buy_price = row[_idx(s, "price")]
                buy_preference = row[_idx(s, "pref")]

                # If we have some empty slots, just buy the asset outright
                if self.active_positions_count < max_active_positions:
                    self.buy_to_open(s, date, buy_price)
                    continue

                # If are holding max_active_positions, evaluate a swap based on
                # preference
                _active = self.active_symbols
                active_prefs = [(s, row[_idx(s, "pref")]) for s in _active]

                _min = min(active_prefs, key=lambda k: k[1])
                min_active_symbol, min_active_preference = _min

                # If a more preferable symbol exists, then sell an old one
                if min_active_preference < buy_preference:
                    sell_price = row[_idx(min_active_symbol, "price")]
                    self.sell_to_close(min_active_symbol, date, sell_price)
                    self.buy_to_open(s, date, buy_price)

            # Update price data everywhere
            for s in self.active_symbols:
                price = row[_idx(s, "price")]
                position = active_positions_by_symbol[s]
                position.record_price_update(date, price)

        # Sell all positions and mark simulation as complete
        for s in self.active_symbols:
            self.sell_to_close(s, date, row[_idx(s, "price")])
        self.portfolio_history.finish()
