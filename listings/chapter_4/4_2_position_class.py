import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple, List, Dict, Callable, NewType, Any
from collections import OrderedDict, defaultdict

from pypm import metrics, signals, data_io

Symbol = NewType('Symbol', str)
Dollars = NewType('Dollars', float)

DATE_FORMAT_STR = '%a %b %d, %Y'
def _pdate(date: pd.Timestamp):
    """Pretty-print a datetime with just the date"""
    return date.strftime(DATE_FORMAT_STR)

class Position(object):
    """
    A simple object to hold and manipulate data related to long stock trades.

    Allows a single buy and sell operation on an asset for a constant number of 
    shares.

    The __init__ method is equivelant to a buy operation. The exit
    method is a sell operation.
    """

    def __init__(self, symbol: Symbol, entry_date: pd.Timestamp, 
        entry_price: Dollars, shares: int):
        """
        Equivelent to buying a certain number of shares of the asset
        """

        # Recorded on initialization
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.symbol = symbol

        # Recorded on position exit
        self.exit_date: pd.Timestamp = None
        self.exit_price: Dollars = None

        # For easily getting current portolio value
        self.last_date: pd.Timestamp = None
        self.last_price: Dollars = None

        # Updated intermediately
        self._dict_series: Dict[pd.Timestamp, Dollars] = OrderedDict()
        self.record_price_update(entry_date, entry_price)

        # Cache control for pd.Series representation
        self._price_series: pd.Series = None
        self._needs_update_pd_series: bool = True

    def exit(self, exit_date, exit_price):
        """
        Equivelent to selling a stock holding
        """
        assert self.entry_date != exit_date, 'Churned a position same-day.'
        assert not self.exit_date, 'Position already closed.'
        self.record_price_update(exit_date, exit_price)
        self.exit_date = exit_date
        self.exit_price = exit_price

    def record_price_update(self, date, price):
        """
        Stateless function to record intermediate prices of existing positions
        """
        self.last_date = date
        self.last_price = price
        self._dict_series[date] = price

        # Invalidate cache on self.price_series
        self._needs_update_pd_series = True

    @property
    def price_series(self) -> pd.Series:
        """
        Returns cached readonly pd.Series 
        """
        if self._needs_update_pd_series or self._price_series is None:
            self._price_series = pd.Series(self._dict_series)
            self._needs_update_pd_series = False
        return self._price_series

    @property
    def last_value(self) -> Dollars:
        return self.last_price * self.shares

    @property
    def is_active(self) -> bool:
        return self.exit_date is None

    @property
    def is_closed(self) -> bool:
        return not self.is_active
    
    @property
    def value_series(self) -> pd.Series:
        """
        Returns the value of the position over time. Ignores self.exit_date.
        Used in calculating the equity curve.
        """
        assert self.is_closed, 'Position must be closed to access this property'
        return self.shares * self.price_series[:-1]

    @property
    def percent_return(self) -> float:
        return (self.exit_price / self.entry_price) - 1
    
    @property
    def entry_value(self) -> Dollars:
        return self.shares * self.entry_price

    @property
    def exit_value(self) -> Dollars:
        return self.shares * self.exit_price

    @property
    def change_in_value(self) -> Dollars:
        return self.exit_value - self.entry_value

    @property
    def trade_length(self):
        return len(self._dict_series) - 1
    
    def print_position_summary(self):
        _entry_date = _pdate(self.entry_date)
        _exit_date = _pdate(self.exit_date)
        _days = self.trade_length

        _entry_price = round(self.entry_price, 2)
        _exit_price = round(self.exit_price, 2)

        _entry_value = round(self.entry_value, 2)
        _exit_value = round(self.exit_value, 2)

        _return = round(100 * self.percent_return, 1)
        _diff = round(self.change_in_value, 2)

        print(f'{self.symbol:<5}     Trade summary')
        print(f'Date:     {_entry_date} -> {_exit_date} [{_days} days]')
        print(f'Price:    ${_entry_price} -> ${_exit_price} [{_return}%]')
        print(f'Value:    ${_entry_value} -> ${_exit_value} [${_diff}]')
        print()

    def __hash__(self):
        """
        A unique position will be defined by a unique combination of an 
        entry_date and symbol, in accordance with our constraints regarding 
        duplicate, variable, and compound positions
        """
        return hash((self.entry_date, self.symbol))