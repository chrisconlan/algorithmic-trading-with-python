import pandas as pd
import datetime

data = {
    "SPY": {
        datetime.date(2000, 1, 4): 100,
        datetime.date(2000, 1, 5): 101,
    },
    "AAPL": {
        datetime.date(2000, 1, 4): 300,
        datetime.date(2000, 1, 5): 303,
    },
}

# Begin listing

# Create a series
series = pd.Series(data=data["SPY"])
print(series)
# Returns ...
# 2000-01-04    100
# 2000-01-05    101
# dtype: int64
