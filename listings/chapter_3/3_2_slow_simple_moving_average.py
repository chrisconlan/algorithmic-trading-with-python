def slow_moving_average(values: List[float], m: int = 20):
    """
    This is O(nm) time, because it re-computes the sum at every step
    1 + 2 + 3 + 4 + ... / m
    2 + 3 + 4 + 5 + ... / m
    3 + 4 + 5 + 6 + ... / m
    4 + 5 + 6 + 7 + ... / m
    and so on ...
    Leading to approx (m-1) * n individual additions.
    """

    # Initial values
    moving_average = [None] * (m - 1)

    for i in range(m - 1, len(values)):
        the_average = np.mean(values[(i - m + 1):i + 1])
        moving_average.append(the_average)

    return moving_average
