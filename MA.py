import yfinance as yf

df = yf.download('SPY')
print(df.head())
