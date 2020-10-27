from pypm import data_io, metrics

df = data_io.load_eod_data("AWU")
print((metrics.calculate_cagr(df["close"])))
