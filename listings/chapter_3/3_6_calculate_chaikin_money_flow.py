def calculate_money_flow_volume_series(df: pd.DataFrame) -> pd.Series:
    """
    Calculates money flow series
    """
    mfv = (df["volume"] * (2 * df["close"] - df["high"] - df["low"]) /
           (df["high"] - df["low"]))
    return mfv


def calculate_money_flow_volume(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """
    Calculates money flow volume, or q_t in our formula
    """
    return calculate_money_flow_volume_series(df).rolling(n).sum()


def calculate_chaikin_money_flow(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """
    Calculates the Chaikin money flow
    """
    return calculate_money_flow_volume(df, n) / df["volume"].rolling(n).sum()
