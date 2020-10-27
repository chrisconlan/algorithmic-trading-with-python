from sklearn.linear_model import LinearRegression


def calculate_pure_profit_score(price_series: pd.Series) -> float:
    """
    Calculates the pure profit score
    """
    cagr = calculate_cagr(price_series)

    # Build a single column for a predictor, t
    t: np.ndarray = np.arange(0, price_series.shape[0]).reshape(-1, 1)

    # Fit the regression
    regression = LinearRegression().fit(t, price_series)

    # Get the r-squared value
    r_squared = regression.score(t, price_series)

    return cagr * r_squared
