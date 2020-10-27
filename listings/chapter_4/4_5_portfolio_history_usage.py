import pandas as pd
from pypm import data_io
from pypm.portfolio import Position, PortfolioHistory

symbol = "AWU"
df = data_io.load_eod_data(symbol)

portfolio_history = PortfolioHistory()
initial_cash = cash = 10000

for i, row in enumerate(df.itertuples()):
    date = row.Index
    price = row.close

    if i == 123:
        # Figure out how many shares to buy
        shares_to_buy = initial_cash / price

        # Record the position
        position = Position(symbol, date, price, shares_to_buy)

        # Spend all of your cash
        cash -= initial_cash

    elif 123 < i < 2345:
        position.record_price_update(date, price)

    elif i == 2345:
        # Sell the asset
        position.exit(date, price)

        # Get your cash back
        cash += price * shares_to_buy

        # Record the position
        portfolio_history.add_to_history(position)

    # Record cash at every step
    portfolio_history.record_cash(date, cash)

portfolio_history.finish()

portfolio_history.print_position_summaries()
# Returns ...
# AWU       Trade summary
# Date:     Wed Jun 30, 2010 -> Tue Apr 30, 2019 [2222 days]
# Price:    $220.34 -> $386.26 [75.3%]
# Value:    $10000.0 -> $17530.18 [$7530.18]

portfolio_history.print_summary()
# Returns ...
# Equity: $17530.18
# Percent Return: 75.30%
# S&P 500 Return: 184.00%
#
# Number of trades: 1
# Average active trades: 1.00
#
# CAGR: 5.78%
# S&P 500 CAGR: 11.02%
# Excess CAGR: -5.24%
#
# Annualized Volatility: 29.97%
# Sharpe Ratio: 0.19
# Jensen's Alpha: -0.000198
#
# Dollar Max Drawdown: $9006.08
# Percent Max Drawdown: 60.08%
# Log Max Drawdown Ratio: -0.36

portfolio_history.plot()
