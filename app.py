import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
import datetime as dt
import plotly.graph_objects as go

plt.style.use('fivethirtyeight')

# No need for %matplotlib inline in .py files

stock = "POWERGRID.NS"
start = dt.datetime(2000 , 1 , 1)
end = dt.datetime(2025 , 6 , 18)

df = yf.download(stock , start , end)
#print(df.head())
#print(df.tail())

print(df.shape)

print(df.info())

print(df.describe())
df = df.reset_index()

df.to_csv("powergrid.csv")
data01 = pd.read_csv("powergrid.csv")


fig = go.Figure(data=[go.Candlestick(x = data01['Date'], open = data01['Open'], high = data01['High'], low = data01['Low'], close = data01['Close'])])
fig.update_layout(xaxis_rangeslider_visible = False)
fig.show()