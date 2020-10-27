class PortfolioHistory(object):
    """
    Holds Position objects and keeps track of portfolio variables.
    Produces summary statistics.
    """

    def __init__(self):
        # Keep track of positions, recorded in this list after close
        self.position_history: List[Position] = []
        self._logged_positions: Set[Position] = set()

        # Keep track of the last seen date
        self.last_date: pd.Timestamp = pd.Timestamp.min

        # Readonly fields
        self._cash_history: Dict[pd.Timestamp, Dollars] = dict()
        self._simulation_finished = False
        self._spy: pd.DataFrame = pd.DataFrame()
        self._spy_log_returns: pd.Series = pd.Series()

    def add_to_history(self, position: Position):
        _log = self._logged_positions
        assert position not in _log, "Recorded the same position twice."
        assert position.is_closed, "Position is not closed."
        self._logged_positions.add(position)
        self.position_history.append(position)
        self.last_date = max(self.last_date, position.last_date)

    def record_cash(self, date, cash):
        self._cash_history[date] = cash
        self.last_date = max(self.last_date, date)

    @staticmethod
    def _as_oseries(d: Dict[pd.Timestamp, Any]) -> pd.Series:
        return pd.Series(d).sort_index()

    def _compute_cash_series(self):
        self._cash_series = self._as_oseries(self._cash_history)

    @property
    def cash_series(self) -> pd.Series:
        return self._cash_series

    def _compute_portfolio_value_series(self):
        value_by_date = defaultdict(float)
        last_date = self.last_date

        # Add up value of assets
        for position in self.position_history:
            for date, value in list(position.value_series.items()):
                value_by_date[date] += value

        # Make sure all dates in cash_series are present
        for date in self.cash_series.index:
            value_by_date[date] += 0

        self._portfolio_value_series = self._as_oseries(value_by_date)

    @property
    def portfolio_value_series(self):
        return self._portfolio_value_series

    def _compute_equity_series(self):
        c_series = self.cash_series
        p_series = self.portfolio_value_series
        assert all(c_series.index == p_series.index
                  ), "portfolio_series has dates not in cash_series"
        self._equity_series = c_series + p_series

    @property
    def equity_series(self):
        return self._equity_series

    def _compute_log_return_series(self):
        self._log_return_series = metrics.calculate_log_return_series(
            self.equity_series)

    @property
    def log_return_series(self):
        return self._log_return_series

    def _assert_finished(self):
        assert self._simulation_finished, (
            "Simuation must be finished by running self.finish() in order " +
            "to access this method or property.")

    def finish(self):
        """
        Notate that the simulation is finished and compute readonly values
        """
        self._simulation_finished = True
        self._compute_cash_series()
        self._compute_portfolio_value_series()
        self._compute_equity_series()
        self._compute_log_return_series()
        self._assert_finished()

    def compute_portfolio_size_series(self) -> pd.Series:
        size_by_date = defaultdict(int)
        for position in self.position_history:
            for date in position.value_series.index:
                size_by_date[date] += 1
        return self._as_oseries(size_by_date)

    @property
    def spy(self):
        if self._spy.empty:
            self._spy = data_io.load_spy_data()
        return self._spy

    @property
    def spy_log_returns(self):
        if self._spy_log_returns.empty:
            close = self.spy["close"]
            self._spy_log_returns = metrics.calculate_log_return_series(close)
        return self._spy_log_returns

    @property
    def percent_return(self):
        return metrics.calculate_percent_return(self.equity_series)

    @property
    def spy_percent_return(self):
        return metrics.calculate_percent_return(self.spy["close"])

    @property
    def cagr(self):
        return metrics.calculate_cagr(self.equity_series)

    @property
    def volatility(self):
        return metrics.calculate_annualized_volatility(self.log_return_series)

    @property
    def sharpe_ratio(self):
        return metrics.calculate_sharpe_ratio(self.equity_series)

    @property
    def spy_cagr(self):
        return metrics.calculate_cagr(self.spy["close"])

    @property
    def excess_cagr(self):
        return self.cagr - self.spy_cagr

    @property
    def jensens_alpha(self):
        return metrics.calculate_jensens_alpha(
            self.log_return_series,
            self.spy_log_returns,
        )

    @property
    def dollar_max_drawdown(self):
        return metrics.calculate_max_drawdown(self.equity_series, "dollar")

    @property
    def percent_max_drawdown(self):
        return metrics.calculate_max_drawdown(self.equity_series, "percent")

    @property
    def log_max_drawdown_ratio(self):
        return metrics.calculate_log_max_drawdown_ratio(self.equity_series)

    @property
    def number_of_trades(self):
        return len(self.position_history)

    @property
    def average_active_trades(self):
        return self.compute_portfolio_size_series().mean()

    @property
    def final_cash(self):
        self._assert_finished()
        return self.cash_series[-1]

    @property
    def final_equity(self):
        self._assert_finished()
        return self.equity_series[-1]

    def print_position_summaries(self):
        for position in self.position_history:
            position.print_position_summary()

    def print_summary(self):
        self._assert_finished()
        s = (f"Equity: ${self.final_equity:.2f}\n"
             f"Percent Return: {100*self.percent_return:.2f}%\n"
             f"S&P 500 Return: {100*self.spy_percent_return:.2f}%\n\n"
             f"Number of trades: {self.number_of_trades}\n"
             f"Average active trades: {self.average_active_trades:.2f}\n\n"
             f"CAGR: {100*self.cagr:.2f}%\n"
             f"S&P 500 CAGR: {100*self.spy_cagr:.2f}%\n"
             f"Excess CAGR: {100*self.excess_cagr:.2f}%\n\n"
             f"Annualized Volatility: {100*self.volatility:.2f}%\n"
             f"Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
             f"Jensen's Alpha: {self.jensens_alpha:.6f}\n\n"
             f"Dollar Max Drawdown: ${self.dollar_max_drawdown:.2f}\n"
             f"Percent Max Drawdown: {100*self.percent_max_drawdown:.2f}%\n"
             f"Log Max Drawdown Ratio: {self.log_max_drawdown_ratio:.2f}\n")

        print(s)

    def plot(self, show=True) -> plt.Figure:
        """
        Plots equity, cash and portfolio value curves.
        """
        self._assert_finished()

        figure, axes = plt.subplots(nrows=3, ncols=1)
        figure.tight_layout(pad=3.0)
        axes[0].plot(self.equity_series)
        axes[0].set_title("Equity")
        axes[0].grid()

        axes[1].plot(self.cash_series)
        axes[1].set_title("Cash")
        axes[1].grid()

        axes[2].plot(self.portfolio_value_series)
        axes[2].set_title("Portfolio Value")
        axes[2].grid()

        if show:
            plt.show()

        return figure
