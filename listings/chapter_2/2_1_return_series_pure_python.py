def calculate_return_series(prices: List[float]) -> List[float]:
    """
    Calculates return series as a parallel list of returns on prices
    """
    return_series = [None]
    for i in range(1, len(prices)):
        return_series.append((prices[i] / prices[i - 1]) - 1)

    return return_series
