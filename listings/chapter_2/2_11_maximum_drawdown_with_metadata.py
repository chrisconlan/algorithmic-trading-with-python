def calculate_max_drawdown_with_metadata(series: pd.Series,
                                         method: str = "log") -> Dict[str, Any]:
    """
    Calculates max_drawdown and stores metadata about when and where. Returns
    a dictionary of the form
        {
            'max_drawdown': float,
            'peak_date': pd.Timestamp,
            'peak_price': float,
            'trough_date': pd.Timestamp,
            'trough_price': float,
        }
    """

    assert (
        method in DRAWDOWN_EVALUATORS
    ), f'Method "{method}" must by one of {list(DRAWDOWN_EVALUATORS.keys())}'

    evaluator = DRAWDOWN_EVALUATORS[method]

    max_drawdown = 0
    local_peak_date = peak_date = trough_date = series.index[0]
    local_peak_price = peak_price = trough_price = series.iloc[0]

    for date, price in series.items():

        # Keep track of the rolling max
        if price > local_peak_price:
            local_peak_date = date
            local_peak_price = price

        # Compute the drawdown
        drawdown = evaluator(price, local_peak_price)

        # Store new max drawdown values
        if drawdown > max_drawdown:
            max_drawdown = drawdown

            peak_date = local_peak_date
            peak_price = local_peak_price

            trough_date = date
            trough_price = price

    return {
        "max_drawdown": max_drawdown,
        "peak_date": peak_date,
        "peak_price": peak_price,
        "trough_date": trough_date,
        "trough_price": trough_price,
    }
