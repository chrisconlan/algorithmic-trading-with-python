def calculate_log_max_drawdown_ratio(series: pd.Series) -> float:
    log_drawdown = calculate_max_drawdown(series, method='log')
    log_return = np.log(series.iloc[-1]) - np.log(series.iloc[0])
    return log_return - log_drawdown