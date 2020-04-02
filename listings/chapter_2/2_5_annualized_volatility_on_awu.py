from pypm import data_io, metrics

df = data_io.load_eod_data('AWU')
return_series = metrics.calculate_log_return_series(df['close'])
print(metrics.calculate_annualized_volatility(return_series))
