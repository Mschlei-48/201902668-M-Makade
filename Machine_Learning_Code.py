import pandas as pd
import pandas_datareader.data as pdr
import datetime as dt

start_date = dt.datetime(2010, 1, 1)
end_date = dt.datetime(2019, 6, 30)
df = pdr.DataReader("MMM", "yahoo", start_date, end_date)
print(df.head())