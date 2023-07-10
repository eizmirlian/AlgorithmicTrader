import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import torch
from torch import nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
from bi_lstm import bi_lstm

device = torch.device('cpu')

def get_data(STOCK_NAMES):
    today_date = datetime.datetime.today()
    delta = datetime.timedelta(days= 1825)
    starting_date = today_date - delta

    start_data = str(starting_date).split()[0]
    end_data = str(today_date).split()[0]
        
            
    stock_data = yf.download(STOCK_NAMES, start= start_data)
    if len(STOCK_NAMES) > 1:
        for ticker in STOCK_NAMES:
            ticker_close_prices = stock_data['Close'][ticker].values
            print(ticker)
            train_model(ticker, ticker_close_prices)
    else:
        ticker_close_prices = stock_data['Close'].values
        print(STOCK_NAMES[0])
        train_model(STOCK_NAMES[0], ticker_close_prices)

def train_model(ticker_name, ticker_dataset):
    NUM_EPOCHS = 10
    LEARNING_RATE = .01

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(ticker_dataset.reshape(-1, 1))
    
    training_length = int(len(scaled_data) * .8)
    training_data = scaled_data[:training_length, :]
    testing_data = scaled_data[training_length :, :]

    x_train = []
    y_train = []
    for i in range(training_length - 60):
        x_train.append(training_data[i : i + 60, :])
        y_train.append(training_data[i + 60, :])
    x_test = []
    y_test = ticker_dataset[training_length + 60 : ]
    for j in range(len(testing_data) - 60):
        x_test.append(testing_data[j : j + 60, :])
    x_train, y_train, x_test, y_test = np.array(x_train).reshape(-1, 60), np.array(y_train).reshape(-1, 1), np.array(x_test).reshape(-1, 60), np.array(y_test)
    x_train, x_test = x_train[:, np.newaxis, :], x_test[:, np.newaxis, :]
    
    print(x_train.shape, len(x_train))
    print(x_test.shape, len(x_test))
    model = bi_lstm(60, 1, 20, 1)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE, weight_decay=1e-5)
    loss_func = torch.nn.MSELoss()

    for epoch in range(NUM_EPOCHS):
        model.train()

        sum_loss = 0
        for input_ind in range(len(x_train)):
            input = torch.tensor(x_train[input_ind, :, :], dtype= torch.float)
            actual = torch.tensor(y_train[input_ind, :], dtype= torch.float).unsqueeze(1)
            optimizer.zero_grad()
            pred = model(input)
            loss = loss_func(pred, actual)
            loss.backward()
            sum_loss += loss.item()
            optimizer.step()
        print("Average Training Loss for Epoch " + str(epoch + 1) + ":", sum_loss / len(x_train))

        model.eval()
        predictions = []
        for test_ind in range(len(x_test)):
            test_input = torch.tensor(x_test[test_ind, :, :], dtype= torch.float)
            prediction = model(test_input)
            predictions.append(prediction.detach().numpy()[0])

        scaled_predictions = scaler.inverse_transform(predictions)

        print("Validation Loss for Epoch " + str(epoch + 1) + ":", rmse(scaled_predictions, y_test))
    torch.save(model, "lstm_models\_" + ticker_name)

def rmse(predicted, actual):
    diff = predicted - actual
    mse = np.mean(diff)**2
    return np.sqrt(mse)

def evaluate_model(ticker):
    today_date = datetime.datetime.today()
    delta = datetime.timedelta(days= 90)
    starting_date = today_date - delta
    start_data = str(starting_date).split()[0]

    stock_data = yf.download([ticker], start= start_data)
    close_prices = stock_data['Close'].values

    scaler = MinMaxScaler()
    test_data = scaler.fit_transform(close_prices.reshape(-1, 1))
    test_input = []

    for j in range(len(test_data) - 60):
        test_input.append(test_data[j : j + 60, :])
    #print(test_input)
    test_input = np.array(test_input).reshape(-1, 60)[:, :, np.newaxis]

    model = keras.models.load_model("lstm_models\_" + ticker)
    predictions = model.predict(test_input)
    scaled_predictions = scaler.inverse_transform(predictions)
    print(scaled_predictions[-1])

    validation = stock_data.filter(['Close'])[60: ]
    validation['Predictions'] = scaled_predictions
    plt.figure(figsize=(16,8))
    plt.title(ticker)
    plt.xlabel('Date')
    plt.ylabel('Close Price USD ($)')
    plt.plot(validation[['Close', 'Predictions']])
    plt.legend(['Actual', 'Predictions'], loc='lower right')
    plt.show()

def predict_future_prices(ticker):
    today_date = datetime.datetime.today()
    delta = datetime.timedelta(days= 90)
    starting_date = today_date - delta
    start_data = str(starting_date).split()[0]

    stock_data = yf.download([ticker], start= start_data)
    close_prices = stock_data['Close'].values

    model = keras.models.load_model("lstm_models\_" + ticker)

    counter = 0
    while counter < 5:
        scaler = MinMaxScaler()
        test_data = scaler.fit_transform(close_prices.reshape(-1, 1))
        test_input = []

        for j in range(len(test_data) - 60):
            test_input.append(test_data[j : j + 60, 0])
            #print(test_input)

        test_input = np.array(test_input)[:, :, np.newaxis]
        
        predictions = model.predict(test_input)
        scaled_predictions = scaler.inverse_transform(predictions)
        print(scaled_predictions[-1])
        close_prices = np.append(close_prices, scaled_predictions[-1])

        counter += 1


get_data(['GOOG'])