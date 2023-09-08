import sys
import seaborn as sns
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QDir
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mp
from statistics import mode
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, GRU
from sklearn.model_selection import train_test_split
from PyQt5 import QtCore
from PyQt5.QtWidgets import QSlider


class DialogApp(QWidget):
    def __init__(self):
        super().__init__()
        self.epochs = 0
        self.current_parameter = ''
        self.file_name = ''
        self.data = ''
        self.resize(450, 200)
        self.button1 = QPushButton('Загрузить файл выборки')
        self.btnPlotOil = QPushButton("Годовая добыча нефти")
        self.btnPlotGas = QPushButton("Годовая добыча газа")
        self.btnPredict = QPushButton('Спрогнозировать параметры добычи')
        self.btnPlotStats = QPushButton("&Параметры выборки")
        self.button1.clicked.connect(self.get_data)
        self.btnPlotOil.clicked.connect(self.plot_of_annual_oil)
        self.btnPlotGas.clicked.connect(self.plot_of_annual_gas)
        self.btnPredict.clicked.connect(self.predict_data)
        self.epochs_choice = QSlider(QtCore.Qt.Horizontal)
        self.epochs_choice.setMinimum(10)
        self.epochs_choice.setMaximum(500)
        self.epochs_choice.setSingleStep(10)
        self.epochs_choice.setValue(255)
        self.epochs_choice.setTickPosition(QSlider.TicksBelow)
        self.epochs_choice.setTickInterval(100)
        layout = QVBoxLayout()
        layout.addWidget(self.button1)
        label_graph = QLabel('Графики')
        label_graph.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(label_graph)
        layout.addWidget(self.btnPlotOil)
        layout.addWidget(self.btnPlotGas)
        self.setLayout(layout)
        choice_of_model_architecture = QComboBox()
        choice_of_model_architecture.addItem('Слой GRU')
        choice_of_model_architecture.addItem('Слой LSTM')
        choice_of_model_architecture.addItem('8 слоев GRU')
        choice_of_model_architecture.activated[str].connect(self.on_activated)
        label = QLabel("Выбор архитектуры нейросети")
        label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(label)
        layout.addWidget(choice_of_model_architecture, alignment=QtCore.Qt.AlignCenter)
        self.setWindowTitle('Прогнозирование параметров добычи')
        label1 = QLabel('Выбор прогнозируемого параметра')
        label1.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(label1)
        choice_of_predicting_parameter = QComboBox()
        choice_of_predicting_parameter.addItem('Год. доб. нефти, тыс.т')
        choice_of_predicting_parameter.addItem('Год. доб. газа, млн.м3')
        choice_of_predicting_parameter.addItem('Обводнен-ность, %')
        choice_of_predicting_parameter.activated[str].connect(self.parameter_choice)
        layout.addWidget(choice_of_predicting_parameter, alignment=QtCore.Qt.AlignCenter)
        label_epochs = QLabel('Выбор количества эпох обучения')
        label_epochs.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(label_epochs)
        self.epochs_choice.valueChanged.connect(self.value_change)
        data_preprocessing = QComboBox()
        label2 = QLabel('Предобработка данных')
        data_preprocessing.addItem('Не использовать предобработку')
        data_preprocessing.addItem('Использовать скользящее среднее')
        data_preprocessing.activated[str].connect(self.preprocessing)
        layout.addWidget(self.epochs_choice)
        layout.addWidget(label2, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(data_preprocessing, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.btnPredict)

    def get_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Открыть файл выборки', r"<Default dir>",
                                                   "*.csv")
        self.file_name = file_name
        self.data = pd.read_csv(self.file_name, sep=';', decimal=',')
        datas_annual_oil_variety = []
        datas_annual_oil_expected_value = []
        datas_annual_oil_moda = []
        datas_annual_oil_scope = []
        datas_variety_coefficient = []
        variety_coefficient = []
        for j in self.data:
            column_variety_coefficient = np.std(self.data[j]) / np.mean(self.data[j])
            if np.std(self.data[j]) != 0.0:
                variety_coefficient.append(column_variety_coefficient)
        datas_variety_coefficient.append(variety_coefficient)
        datas_annual_oil_variety.append(np.std(self.data['Год. доб. нефти, тыс.т']))
        datas_annual_oil_expected_value.append(np.var(self.data['Год. доб. нефти, тыс.т']))
        datas_annual_oil_moda.append(mode(self.data['Год. доб. нефти, тыс.т']))
        datas_annual_oil_scope.append(np.max(self.data['Год. доб. нефти, тыс.т']) -
                                      np.min(self.data['Год. доб. нефти, тыс.т']))

    def plot_of_annual_oil(self):
        plt.scatter(self.data['Годы'], self.data['Год. доб. нефти, тыс.т'])
        plt.xlabel('Годы')
        plt.ylabel('Год. доб. нефти, тыс.т')
        plt.title(self.file_name[52:].replace('.csv', ''))
        plt.show()

    def plot_of_annual_gas(self):
        listing = self.data['Год. доб. газа, млн.м3']
        plt.hist(listing, orientation="horizontal")
        plt.title(self.file_name[52:].replace('.csv', ''))
        plt.xlabel('Годы')
        plt.ylabel('Год. доб. газа, млн.м3')
        plt.show()
        # df = self.data.pivot_table(index='Обводнен-ность, %', columns='Год. доб. нефти, тыс.т')
        # sns.heatmap(df)
        # plt.show()

    def on_activated(self, idx):
        self.current_architecture = idx

    def predict_data(self):
        x = self.data.values
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x)
        data = pd.DataFrame(x_scaled)
        print(data)
        # data[1] == Y
        # Y is being predicted, based on X (in this case oil based on years)
        # data[2] == X
        X = data.loc[:, 1]
        if self.current_parameter == 'Год. доб. нефти, тыс.т':
            X = data.loc[:, 0]
            Y = data.loc[:, 1]
        elif self.current_parameter == 'Год. доб. газа, млн.м3':
            Y = data.loc[:, 5]
        else:
            Y = data.loc[:, 10]
        print(X)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
        model = Sequential()
        model.add(Dense(64, input_shape=(14, 1)))
        learning_rate = 0.0001
        if self.current_architecture == 'Слой LSTM':
            model.add(LSTM(300))
        elif self.current_architecture == 'Слой GRU':
            model.add(GRU(300))
        else:
            for i in range(7):
                model.add(GRU(300, input_shape=(300, ), return_sequences=True))
        model.add(Activation('relu'))
        model.add(Dense(1))
        decay_rate = learning_rate / self.epochs
        model.compile(loss='mse', optimizer='adam')
        estimation_object = model.fit(X_train, Y_train, epochs=self.epochs, batch_size=32, verbose=1,
                                      validation_split=0.25)
        predict = model.predict(X_test)
        predicted = predict.flatten()
        original = Y_test.values
        plt.figure(1)
        plt.plot(predicted, color='blue', label='Predicted data')
        plt.plot(original, color='red', label='Original data')
        plt.legend(loc='best')
        plt.title('Actual and predicted')
        loss = []
        for i in estimation_object.history['loss']:
            loss.append(i)
        plt.figure(2)
        plt.plot(loss)
        plt.xlabel('MSE')
        predicted_and_original_difference = []
        for i, j in enumerate(original):
            predicted_and_original_difference.append(abs(predicted[i] - j))
        print(np.mean(predicted_and_original_difference))
        loss = []
        for i in estimation_object.history['loss']:
            loss.append(i)
        print(np.min(loss))
        model.save(self.current_architecture+'/'+self.current_parameter)
        plt.show()

    def parameter_choice(self, parameter):
        self.current_parameter = parameter

    def value_change(self):
        self.epochs = self.epochs_choice.value()

    def preprocessing(self, choice):
        if choice == 'Использовать скользящее среднее':
            data = self.data
            for num, _ in enumerate(data[self.current_parameter]):
                t = []
                n = 3  # скользящее среднее по n точкам
                shift = (n - 1) // 2
                for i in range(n):
                    index = num - shift + i
                    if index >= len(data[self.current_parameter]):
                        t.append(data[self.current_parameter][index % len(data[self.current_parameter])])
                    elif index >= 0:
                        t.append(data[self.current_parameter][index])
                    else:
                        t.append(data[self.current_parameter][len(data[self.current_parameter]) + index])
                data[self.current_parameter][num] = np.sum(t) / n
            self.data = data
        else:
            pass


app = QApplication(sys.argv)
demo = DialogApp()
# demo.get_data()
# demo.on_activated('Слой GRU')
# demo.parameter_choice('Год. доб. нефти, тыс.т')
# demo.value_change()
# demo.preprocessing('Использовать скользящее среднее')
# demo.predict_data()
demo.show()
sys.exit(app.exec_())
