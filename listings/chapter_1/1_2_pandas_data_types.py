import pandas as pd
import datetime

data = {
    'SPY': {
        datetime.date(2000, 1, 4): 100,
        datetime.date(2000, 1, 5): 101,
    },
    'AAPL': {
        datetime.date(2000, 1, 4): 300,
        datetime.date(2000, 1, 5): 303,
    },
}
df: pd.DataFrame = pd.DataFrame(data=data)
print(df)
# Returns ...
#             SPY  AAPL
# 2000-01-04  100   300
# 2000-01-05  101   303

# Index by column
aapl_series: pd.Series = df['AAPL']
print(aapl_series)
# Returns ... 
# 2000-01-04    300
# 2000-01-05    303
# Name: AAPL, dtype: int64

# Index by row
start_of_year_row: pd.Series = df.loc[datetime.date(2000, 1, 4)]
print(start_of_year_row)
# Returns ... 
# SPY     100
# AAPL    300
# Name: 2000-01-04, dtype: int64

# Index by both
start_of_year_price: pd.Series = df['AAPL'][datetime.date(2000, 1, 4)]
print(start_of_year_price)
# Returns ... 
# 300