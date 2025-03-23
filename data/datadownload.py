import yfinance as yf
#df = yf.download("SPY", start="2018-01-01", end="2024-01-01")
#df.to_csv("data/SPY_5yr.csv")

# Download the data for the S&P 500 index


# Download the data for the NASDAQ index
df = yf.download("^IXIC", start="2018-01-01", end="2024-01-01")
df.to_csv("data/NASDAQ_5yr.csv")

# Download the data for the Russell 2000 index
df = yf.download("^RUT", start="2018-01-01", end="2024-01-01")
df.to_csv("data/RUT_5yr.csv")

# Download the data for the Dow Jones index
df = yf.download("^DJI", start="2018-01-01", end="2024-01-01")
df.to_csv("data/DJI_5yr.csv")

# Download the data for the Nasdaq 100 index
df = yf.download("^NDX", start="2018-01-01", end="2024-01-01")
df.to_csv("data/NDX_5yr.csv")

# Download the data for the Russell 3000 index
df = yf.download("^RUI", start="2018-01-01", end="2024-01-01")
df.to_csv("data/RUI_5yr.csv")

# Download the data for the Russell 1000 index
df = yf.download("^RUI1000", start="2018-01-01", end="2024-01-01")
df.to_csv("data/RUI1000_5yr.csv")

