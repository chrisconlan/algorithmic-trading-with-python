def fast_moving_average(values: List[float], m: int = 20):
    """
    This is O(n) time, because it keeps track of the intermediate sum.
    Leading to approx 2n individual additions.
    """

    # Initial values
    moving_average = [None] * (m - 1)
    accumulator = sum(values[:m])
    moving_average.append(accumulator / m)

    for i in range(m, len(values)):
        accumulator -= values[i - m]
        accumulator += values[i]
        moving_average.append(accumulator / m)

    return moving_average
