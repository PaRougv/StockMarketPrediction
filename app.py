import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
import datetime as dt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

plt.style.use('fivethirtyeight')

# No need for %matplotlib inline in .py files

stock = "POWERGRID.NS"
start = dt.datetime(2000 , 1 , 1)
end = dt.datetime(2025 , 6 , 18)

df = yf.download(stock , start , end)
#print(df.head())
#print(df.tail())

#print(df.shape)

#print(df.info())

#print(df.describe())
df = df.reset_index()

df.to_csv("powergrid.csv")
data01 = pd.read_csv("powergrid.csv")


fig = go.Figure(data=[go.Candlestick(x = data01['Date'], open = data01['Open'], high = data01['High'], low = data01['Low'], close = data01['Close'])])
fig.update_layout(xaxis_rangeslider_visible = False)
fig.show()

#df.drop(['Date' , 'Adj Close'], axis = 1)

#print(df.columns)

movingavg100 = df.Close.rolling(100).mean()
movingavg200 = df.Close.rolling(200).mean()


# plt.figure(figsize = (12 , 6))
# plt.plot(df['Close'], label = f'{stock} Closing over time', linewidth = 1)
# plt.plot(movingavg100, label = f'{stock} Moving Average Over 100 Values', linewidth = 1)
# plt.plot(movingavg200, label = f'{stock} Moving Average Over 200 Values', linewidth = 1)
# plt.title("Closing of Stockes over time Graph")
# plt.xlabel("Date")
# plt.ylabel("Closing Price")
# plt.legend()
# plt.show()


ema100 = df['Close'].ewm(span = 100 , adjust = False).mean()
ema200 = df['Close'].ewm(span = 200 , adjust = False).mean()

# plt.figure(figsize = (12 , 6))
# plt.plot(df['Close'], label = f'{stock} Closing over time', linewidth = 1)
# plt.plot(ema100, label = f'{stock} Moving Average Over 100 Values', linewidth = 1)
# plt.plot(ema200, label = f'{stock} Moving Average Over 200 Values', linewidth = 1)
# plt.title("Closing of Stockes over time Graph")
# plt.xlabel("Date")
# plt.ylabel("Closing Price")
# plt.legend()
# plt.show()

data_testing = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_training = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

# print(data_testing.shape)
# print(data_training.shape)


scaler = MinMaxScaler(feature_range=(-1,1))
data_training_array = scaler.fit_transform(data_training)
print(data_training_array.shape)
print(data_training_array)

x_train = []
y_train = []


for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100 : i])
    y_train.append(data_training_array[i,0])

x_train , y_train = np.array(x_train) , np.array(y_train)

print(x_train.shape)

model = Sequential()

model.add(LSTM(units = 50 , activation='relu', return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60 , activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units = 80 , activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units = 120 , activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 1))

#print(model.summary())

model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(x_train, y_train, epochs = 50)