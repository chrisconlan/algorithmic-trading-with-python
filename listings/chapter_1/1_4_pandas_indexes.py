import pandas as pd
import datetime

### Begin listing

dates = [datetime.date(2000, 1, i) for i in range(1, 11)]
values = [i**2 for i in range(1, 11)]
series = pd.Series(data=values, index=dates)

# O(n) time complexity search through a list
print(datetime.date(2000, 1, 5) in dates)
# Returns ...
# True

# O(1) time complexity search through an index
print(datetime.date(2000, 1, 5) in series.index)
# Returns ...
# True
