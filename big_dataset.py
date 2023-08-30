import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout
import seaborn as sns
import keras

data = pd.read_csv('pressure_dataset_2.csv', sep=';', decimal=',')[:-1]
print(data)
# mean_gas = np.mean(data['Дебит газа'])
# for num, i in enumerate(data['Дебит газа']):
#     data['Дебит газа'][num] = np.abs(i - mean_gas)
# print(data)
for num, i in enumerate(data['Время сбора данных']):
    t = i.split(':')
    t = [int(i) for i in t]
    time_of_measurement_in_seconds = t[0] * 3600 + t[1] * 60 + t[2]
    data['Время сбора данных'][num] = time_of_measurement_in_seconds
plt.plot(data['Время сбора данных'], data['Давление'])
for num, _ in enumerate(data['Давление']):
    t = []
    n = 3 # скользящее среднее по n точкам
    shift = (n - 1) // 2
    for i in range(n):
        index = num - shift + i
        if index >= len(data['Давление']):
            t.append(data['Давление'][index % len(data['Давление'])])
        elif index >= 0:
            t.append(data['Давление'][index])
        else:
            t.append(data['Давление'][len(data['Давление']) + index])
    data['Давление'][num] = np.sum(t) / n
x = data.values
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
data = pd.DataFrame(x_scaled)
# print(data)
# data[1] == Y
# Y is being predicted, based on X (in this case oil based on years)
# data[2] == X
X = data.loc[:, 2]
Y = data.loc[:, 1]
# print(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
model = Sequential()
model.add(Dense(64, input_shape=(14, 1)))
model.add(GRU(300, return_sequences=True))
# model.add(GRU(300))
model.add(Dense(1))
epochs = 10
learning_rate = 0.001
decay_rate = learning_rate / epochs
model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=learning_rate, decay=decay_rate))
estimation_object = model.fit(X_train, Y_train, epochs=epochs, batch_size=32, verbose=1,
                              validation_split=0.25)
predict = model.predict(X_test)
predicted = predict.flatten()
original = Y_test.values
plt.figure(2)
plt.plot(predicted, color='blue', label='Predicted data')
plt.plot(original, color='red', label='Original data')
plt.legend(loc='best')
plt.title('Actual and predicted')
name_of_fig = f'plots/epochs_{str(epochs)}.png'
plt.savefig(name_of_fig)
model.summary()
predicted_and_original_difference = []
for i, j in enumerate(original):
    predicted_and_original_difference.append(abs(predicted[i] - j))
print(np.mean(predicted_and_original_difference))
loss = []
for i in estimation_object.history['loss']:
    loss.append(i)
plt.figure(3)
plt.plot(loss)
plt.xlabel('MSE')
plt.show()
