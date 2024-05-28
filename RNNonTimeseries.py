import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.python.keras.layers import Dense, LSTM
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')

df = pd.read_csv('RNNonTimesries\RSCCASN.csv')
# print(df)
# print(df.info())
df = pd.read_csv('RNNonTimesries\RSCCASN.csv', parse_dates=True, index_col='DATE')
# print(df)
# print(df.info())

# df.plot()
# plt.show()
# print(len(df))
# print(len(df) - 18)

test_size = 18
test_ind = len(df) - test_size
train = df.iloc[:test_ind]
test = df.iloc[test_ind:]

scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)
print(train.shape)

# print(len(test))

length = 12
batch_size = 1

generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=1)
# print(len(generator))
x, y = generator[0]
print(x)
print(x.shape)
print(y)
print(y.shape)

x_train = []
y_train = []

for x, y in generator:
    x_train.append(x)
    y_train.append(y)

x_train = np.array(x_train).reshape(304, 12, 1)
y_train = np.array(y_train).reshape(304, 1, 1)

n_feature = 1

model = Sequential()

model.add(LSTM(750,activation='relu', input_shape=(length, n_feature)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

print(model.summary())

early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=2)

x_test = []
y_test = []

validation_generator = TimeseriesGenerator(scaled_test, scaled_test, length=length, batch_size=1)
# print(len(validation_generator))

for x, y in validation_generator:
    x_test.append(x)
    y_test.append(y)

x_test = np.array(x_test).reshape(6, 12, 1)
y_test = np.array(y_test).reshape(6, 1, 1)

# hist = model.fit(x=x_train, y=y_train, epochs=20, validation_data=(x_test, y_test), callbacks=[early_stop])

# losses = pd.DataFrame(hist.history)
# losses.plot()
# plt.show()
# losses.to_csv('RNNonTimesries\RNNonTimeserie.csv', index=False)

# model.save('RNNonTimesries\RNNonTimeserie.h5')

losses = pd.read_csv('RNNonTimesries\RNNonTimeseriesipynb.csv')
later_model = load_model('RNNonTimesries\RNNonTimeseriesipynb.h5', compile=False)

losses.plot()
plt.show()

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_feature))

test_prediction = []

for i in range(len(test)):
    current_pred = later_model.predict(current_batch)
    test_prediction.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:], [current_pred], axis=1)

# print(len(test_prediction))
test_prediction = scaler.inverse_transform(np.array(test_prediction).reshape(18, 1))
test['Prediction'] = test_prediction
print(test)

test.plot()
plt.show()

# not a very good model
