import pandas as pd
from pypm import data_io, portfolio

symbol = 'AWU'
df = data_io.load_eod_data(symbol)
shares_to_buy = 50

for i, row in enumerate(df.itertuples()):
	date = row.Index
	price = row.close

	if i == 123:
		position = portfolio.Position(symbol, date, price, shares_to_buy)
	elif 123 < i < 234:
		position.record_price_update(date, price)
	elif i == 234:
		position.exit(date, price)

position.print_position_summary()

# Returns ...
# AWU       Trade summary
# Date:     Wed Jun 30, 2010 -> Tue Dec 07, 2010 [111 days]
# Price:    $220.34 -> $305.98 [38.9%]
# Value:    $11017.0 -> $15299.0 [$4282.0]