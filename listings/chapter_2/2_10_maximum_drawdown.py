from typing import Dict, Any, Callable

DRAWDOWN_EVALUATORS: Dict[str, Callable] = {
    "dollar": lambda price, peak: peak - price,
    "percent": lambda price, peak: -((price / peak) - 1),
    "log": lambda price, peak: np.log(peak) - np.log(price),
}


def calculate_drawdown_series(series: pd.Series,
                              method: str = "log") -> pd.Series:
    """
    Returns the drawdown series
    """
    assert (
        method in DRAWDOWN_EVALUATORS
    ), f'Method "{method}" must by one of {list(DRAWDOWN_EVALUATORS.keys())}'

    evaluator = DRAWDOWN_EVALUATORS[method]
    return evaluator(series, series.cummax())


def calculate_max_drawdown(series: pd.Series, method: str = "log") -> float:
    """
    Simply returns the max drawdown as a float
    """
    return calculate_drawdown_series(series, method).max()
