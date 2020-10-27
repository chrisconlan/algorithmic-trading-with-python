from typing import List, Dict, Tuple, Any
import numpy as np
import datetime

# A list of floating point numbers
v: List[float] = [i * 1.23 for i in range(10)]

# A list of mixed type values
v: List[Any] = ["apple", 123, "banana", None]

# A dictionary of floats indexed by dates
v: Dict[datetime.date, float] = {
    datetime.date.today(): 123.456,
    datetime.date(2000, 1, 1): 234.567,
}

# A dictionary of lists of strings indexed by tuples of integers
v: Dict[Tuple[int, int], List[str]] = {
    (2, 3): [
        "apple",
        "banana",
    ],
    (4, 7): [
        "orange",
        "pineapple",
    ],
}

# An incorrect type hint
# Your compiler or IDE might complain about this
v: List[str] = [1, 2, 3]

# A possibly incorrect type hint
# There is no concensus on whether or not this is correct
v: List[float] = [1, None, 3, None, 5]

# This is non-descript but correct
v: List = [(1, 2, "a"), (4, 5, "b")]

# This is more descriptive
v: List[Tuple[int, int, str]] = [(1, 2, "a"), (4, 5, "b")]

# Custom types are supported
from typing import NewType

StockTicker = NewType("StockTicker", np.float64)
ticker: StockTicker = "AAPL"

# Functions can define input and return types


def convert_to_string(value: Any) -> str:
    return str(value)
