def calculate_jensens_alpha(return_series: pd.Series, 
    benchmark_return_series: pd.Series) -> float: 
    """
    Calculates jensens alpha. Prefers input series have the same index. Handles
    NAs.
    """

    # Join series along date index and purge NAs
    df = pd.concat([return_series, benchmark_return_series], sort=True, axis=1)
    df = df.dropna()

    # Get the appropriate data structure for scikit learn
    clean_returns: pd.Series = df[return_series.name]
    clean_benchmarks = pd.DataFrame(df[benchmark_return_series.name])

    # Fit a linear regression and return the alpha
    regression = LinearRegression().fit(clean_benchmarks, y=clean_returns)
    return regression.intercept_