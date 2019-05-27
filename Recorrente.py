# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 07:31:54 2018

@author: Gustavo
"""
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error
import matplotlib.dates as dates

dados = pd.read_csv('petr4.csv')
dados = dados.dropna()
dados_treinamento = dados.iloc[:, 4:5].values


normalizador = MinMaxScaler(feature_range=(0,1))
dados_treinamento_normalizada = normalizador.fit_transform(dados_treinamento)

previsores = []
preco_real = []
for i in range(90, len(dados)):
    previsores.append(dados_treinamento_normalizada[i-90:i, 0])
    preco_real.append(dados_treinamento_normalizada[i, 0])
previsores, preco_real = np.array(previsores), np.array(preco_real)
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

regressor = Sequential()
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50))

regressor.add(Dense(1))

regressor.compile(optimizer = 'adam', loss = 'mae')
regressor.fit(previsores, preco_real, epochs = 300, batch_size = 100,shuffle = False)

dados_teste = pd.read_csv('petr418.csv')
preco_real_teste = dados_teste.iloc[:, 4:5].values
dados_completa = pd.concat((dados['Open'], dados_teste['Open']), axis = 0)
entradas = dados_completa[len(dados_completa) - len(dados_teste) - 90:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

X_teste = []
for i in range(90, len(entradas)):
    X_teste.append(entradas[i-90:i, 0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))
previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)

rmse = sqrt(mean_squared_error(preco_real_teste, previsoes))
print(rmse)

dados_teste['Date'] = pd.to_datetime(dados_teste['Date'])

fig, ax = plt.subplots()
ax.plot_date(dados_teste['Date'],previsoes,'b-',label = "previsto")
ax.plot_date(dados_teste['Date'],preco_real_teste,'r-', label = "real")
ax.xaxis.grid(True, which="minor")
ax.yaxis.grid()
plt.legend(loc='best')
plt.title('Previsão LSTM')
plt.xlabel('DIAS')
plt.ylabel('FECHAMENTO')

ax.xaxis.set_minor_locator(dates.DayLocator(bymonthday=range(1, 32, 5)))
plt.show()