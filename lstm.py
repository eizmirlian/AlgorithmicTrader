import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import torch
import torch.utils.data as data
from torch import nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
from bi_lstm import bi_lstm

device = torch.device('cpu')

#Hyperparameters
LOOK_BACK = 80
BATCH_SIZE = 30
HIDDEN_SIZE = 30
NUM_EPOCHS = 30
LEARNING_RATE = .005


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

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(ticker_dataset.reshape(-1, 1))
    #scaled_data = ticker_dataset.reshape(-1, 1)
    
    training_length = int(len(scaled_data) * .8)
    training_data = scaled_data[:training_length, :]
    testing_data = scaled_data[training_length :, :]

    x_train = []
    y_train = []
    for i in range(training_length - LOOK_BACK):
        x_train.append(training_data[i : i + LOOK_BACK, :])
        y_train.append(training_data[i + LOOK_BACK, :])
    x_test = []
    y_test = ticker_dataset[training_length + LOOK_BACK : ]
    for j in range(len(testing_data) - LOOK_BACK):
        x_test.append(testing_data[j : j + LOOK_BACK, :])
    x_train, y_train, x_test, y_test = np.array(x_train).reshape(-1, LOOK_BACK), np.array(y_train).reshape(-1, 1), np.array(x_test).reshape(-1, LOOK_BACK), np.array(y_test)
    x_train, x_test, y_train, y_test = torch.tensor(x_train, dtype= torch.float), torch.tensor(x_test, dtype= torch.float), torch.tensor(y_train, dtype = torch.float), torch.tensor(y_test, dtype= torch.float)

    train_loader = data.DataLoader(data.TensorDataset(x_train, y_train), shuffle= True, batch_size= BATCH_SIZE)
    test_loader = data.DataLoader(data.TensorDataset(x_test, y_test), shuffle= True, batch_size= BATCH_SIZE)
    
    model = bi_lstm(LOOK_BACK, BATCH_SIZE, HIDDEN_SIZE, 1)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE, weight_decay=1e-5)
    loss_func = torch.nn.MSELoss()

    for epoch in range(NUM_EPOCHS):
        model.train()

        sum_loss = 0
        for X, y in train_loader:
            y_pred = model(X)
            final_x = []

            for batch in X:
                final_x.append(batch[-1].item())
            final_x = torch.tensor(final_x, dtype= torch.float)
            predicted_diff = (y_pred[:, 0] - final_x)
            actual_diff = (y[:, 0] - final_x)

            loss = loss_func(predicted_diff, actual_diff)
            optimizer.zero_grad()
            loss.backward()
            sum_loss += loss.item()
            optimizer.step()
        print("Average Training Loss for Epoch " + str(epoch + 1) + ":", sum_loss / len(x_train))

        model.eval()
        sum_err = 0
        count = 0
        with torch.no_grad():
            for X, y in test_loader:
                prediction = model(X)
                scaled_prediction = torch.tensor(scaler.inverse_transform(prediction.detach().numpy()))
                sum_err += rmse(scaled_prediction, y)
                count += 1
        
        #scaled_predictions = predictions

        print("Validation Loss for Epoch " + str(epoch + 1) + ":", str(sum_err / count))
    torch.save(model.state_dict(), "lstm_models\_" + ticker_name)

def rmse(predicted, actual):
    diff = predicted - actual
    mse = torch.mean(diff)**2
    return torch.sqrt(mse)

def evaluate_model(ticker):
    today_date = datetime.datetime.today()
    delta = datetime.timedelta(days= 140)
    starting_date = today_date - delta
    start_data = str(starting_date).split()[0]

    stock_data = yf.download([ticker], start= start_data)
    close_prices = stock_data['Close'].values

    scaler = MinMaxScaler()
    test_data = scaler.fit_transform(close_prices.reshape(-1, 1))
    #test_data = close_prices.reshape(-1, 1)
    test_input = []

    for j in range(len(test_data) - LOOK_BACK):
        test_input.append(test_data[j : j + LOOK_BACK, :])
    #print(test_input)
    test_input = torch.tensor(np.array(test_input).reshape(-1, LOOK_BACK), dtype = torch.float)
    y_actual = torch.tensor(test_data[LOOK_BACK: ], dtype = torch.float)

    test_loader = data.DataLoader(data.TensorDataset(test_input, y_actual), shuffle= True, batch_size= BATCH_SIZE)

    model = bi_lstm(LOOK_BACK, BATCH_SIZE, HIDDEN_SIZE, 1)
    model.load_state_dict(torch.load("lstm_models\_" + ticker))
    model.eval()

    predictions = []
    for X, y in test_loader:
        prediction = model(X)
        predictions.extend(prediction.detach().numpy())

    scaled_predictions = scaler.inverse_transform(predictions)
    #scaled_predictions = predictions
    #print(scaled_predictions)

    validation = stock_data.filter(['Close'])[LOOK_BACK: ]
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
    delta = datetime.timedelta(days= LOOK_BACK * 2 + BATCH_SIZE)
    starting_date = today_date - delta
    start_data = str(starting_date).split()[0]

    stock_data = yf.download([ticker], start= start_data)
    close_prices = stock_data['Close'].values

    model = bi_lstm(LOOK_BACK, BATCH_SIZE, HIDDEN_SIZE, 1)
    model.load_state_dict(torch.load("lstm_models\_" + ticker))
    model.eval()

    with torch.no_grad():
        counter = 0
        while counter < 5:

            scaler = MinMaxScaler()
            test_data = scaler.fit_transform(close_prices.reshape(-1, 1))

            i = BATCH_SIZE - 1
            pred_input = []
            while i >= 0:
                pred_input.append(test_data[len(test_data) - LOOK_BACK - i : len(test_data) - i, :])
                i -= 1

            pred_input = np.array(pred_input).reshape(-1, LOOK_BACK)
            pred_input = torch.tensor(pred_input, dtype= torch.float)
            pred_loader = data.DataLoader(data.TensorDataset(pred_input, pred_input), batch_size= BATCH_SIZE)

            sampled_predictions = []
            for i in range(100):
                model.dropout_on()

                for X, y in pred_loader:
                    prediction = model(X)
                    prediction = prediction.detach().numpy()

                scaled_predictions = scaler.inverse_transform(prediction)
                #print(scaled_predictions[-1])
                sampled_predictions.append(scaled_predictions[-1][0])

            print(sampled_predictions, np.mean(sampled_predictions))


            close_prices = np.append(close_prices, np.mean(sampled_predictions))
            model.dropout_off()

            counter += 1

get_data(['GOOG'])