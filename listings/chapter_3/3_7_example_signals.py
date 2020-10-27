def create_macd_signal(series: pd.Series,
                       n1: int = 5,
                       n2: int = 34) -> pd.Series:
    """
    Create a momentum-based signal based on the MACD crossover principle.
    Generate a buy signal when the MACD cross above zero, and a sell signal when
    it crosses below zero.
    """

    # Calculate the macd and get the signs of the values.
    macd = calculate_macd_oscillator(series, n1, n2)
    macd_sign = np.sign(macd)

    # Create a copy shifted by some amount.
    macd_shifted_sign = macd_sign.shift(1, axis=0)

    # Multiply by the sign by the boolean. This will have the effect of casting
    # the boolean to an integer (either 0 or 1) and then multiply by the sign
    # (either -1, 0 or 1).
    return macd_sign * (macd_sign != macd_shifted_sign)


def create_bollinger_band_signal(series: pd.Series, n: int = 20) -> pd.Series:
    """
    Create a reversal-based signal based on the upper and lower bands of the
    Bollinger bands. Generate a buy signal when the price is below the lower
    band, and a sell signal when the price is above the upper band.
    """
    bollinger_bands = calculate_bollinger_bands(series, n)
    sell = series > bollinger_bands["upper"]
    buy = series < bollinger_bands["lower"]
    return 1 * buy - 1 * sell
