import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
import datetime as dt

plt.style.use('fivethirtyeight')

# No need for %matplotlib inline in .py files

stock = "POWERGRID.NS"
start = dt.datetime(2000 , 1 , 1)
end = dt.datetime(2025 , 6 , 18)

df = yf.download(stock , start , end)
