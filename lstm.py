import math
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import torch.optim.lr_scheduler as lrs
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
from lstm_toolkit import bi_lstm

device = torch.device('cpu')

#Hyperparameters for the LSTM models trained for each individual stock
LOOK_BACK = 80
BATCH_SIZE = 30
HIDDEN_SIZE = 30
NUM_EPOCHS = 150
NUM_LAYERS = 3
LEARNING_RATE = .01

def get_models(STOCK_NAMES, end_date= datetime.datetime.today()):

    """Creates a dictionary of the requested models using train_model. Can be used to create models trained on older data to simulate what past predictions would look like. \n
    Parameters: \n STOCK_NAMES : (List[Str]) A list of stocks which will have models trained on their historical closing prices.
    \n end_date: (datetime.date object) An optional end date for the data the models are trained on (inclusive)"""

    model_zoo = {}
    current = end_date == datetime.datetime.today()

    # Dates used to download the correct range of stock data. It is set to download and train the LSTM models on the 5 year price history
    delta = datetime.timedelta(days= 1825)
    day_delta = datetime.timedelta(days= 1)
    starting_date = end_date - delta

    start_data = str(starting_date).split()[0]
    end_data = str(end_date + day_delta).split()[0]

    # Downloads stock price history from yfinance for the passed in stock names and the range of dates being the 5 years preceding the end date (inclusive)
    stock_data = yf.download(STOCK_NAMES, start= start_data, end= end_data)

    # runs train_model on each of the stock names passed in, passing in the downloaded dataset
    if len(STOCK_NAMES) > 1:
        for ticker in STOCK_NAMES:
            ticker_close_prices = stock_data['Close'][ticker].values
            print(ticker)
            model_zoo[ticker] = train_model(ticker, ticker_close_prices, current)
    else:
        ticker_close_prices = stock_data['Close'].values
        print(STOCK_NAMES[0])
        model_zoo[STOCK_NAMES[0]] = train_model(STOCK_NAMES[0], ticker_close_prices, current)

    return model_zoo


def train_model(ticker_name, ticker_dataset, current):

    """Trains a custom pytorch Bidirectional LSTM model (more details in lstm_toolkit.py) for the passed in stock name using the passed in data. It is trained to predict the next day closing price given an observation window of length LOOK_BACK. The LSTMs use the Adam optimizer and a Cosine Annealing Learning Rate Scheduler. 
    The loss during training is calculated as the MSE of price changes from the last day of the prediction window to the predicted day.\n
    Parameters: \n ticker_name : (str) The stock name for which the LSTM model will be trained \n ticker_dataset: (numpy array object of floats) The dataset on which the model will be trained"""

    # Scale and reshape the dataset using a sklearn MinMax scaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(ticker_dataset.reshape(-1, 1))

    # Separate the training and testing data
    # NOTE this was changed to train the LSTM model on the full dataset. This was done to ensure that the model was trained on the most recent data, which is the most relevant.
    # There still is some "validation" to see how well the model performs on its training data, but this cannot be called proper testing. Since this process is usually automated anyways, having a proper validation set was considered lower priority
    training_length = len(scaled_data)
    training_data = scaled_data
    testing_data = scaled_data[int(training_length * .8) :, :]

    # Format the training inputs into a series of windows of closing prices of length LOOK_BACK, and the expected output being the next day's closing price. Also does this with a small set of "validation data" (which it's not really because there is an overlap with the training data)
    x_train = []
    y_train = []
    for i in range(training_length - LOOK_BACK):
        x_train.append(training_data[i : i + LOOK_BACK, :])
        y_train.append(training_data[i + LOOK_BACK, :])
    x_test = []
    y_test = ticker_dataset[int(training_length * .8) + LOOK_BACK : ]
    for j in range(len(testing_data) - LOOK_BACK):
        x_test.append(testing_data[j : j + LOOK_BACK, :])
    x_train, y_train, x_test, y_test = np.array(x_train).reshape(-1, LOOK_BACK), np.array(y_train).reshape(-1, 1), np.array(x_test).reshape(-1, LOOK_BACK), np.array(y_test)
    x_train, x_test, y_train, y_test = torch.tensor(x_train, dtype= torch.float), torch.tensor(x_test, dtype= torch.float), torch.tensor(y_train, dtype = torch.float), torch.tensor(y_test, dtype= torch.float)

    # Creates data loaders for the training and testing data with the batch size being BATCH_SIZE
    train_loader = data.DataLoader(data.TensorDataset(x_train, y_train), shuffle= True, batch_size= BATCH_SIZE)
    test_loader = data.DataLoader(data.TensorDataset(x_test, y_test), shuffle= True, batch_size= BATCH_SIZE)
    
    # Creates the Bidirectional LSTM model as an instance of a custom class, passes in the hyperparameters defined above
    model = bi_lstm(LOOK_BACK, BATCH_SIZE, HIDDEN_SIZE, 1, NUM_LAYERS)

    model.to(device)

    # Creates the Adam optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr= LEARNING_RATE, weight_decay=1e-5)
    scheduler = lrs.CosineAnnealingLR(optimizer, 50, .00001)
    
    #Defines the loss function as a mean-square error loss function
    loss_func = torch.nn.MSELoss()

    # Faciltates NUM_EPOCHs of training
    for epoch in range(NUM_EPOCHS):
        # put the model in training mode
        model.train()

        sum_loss = 0
        # iterate through batches of the training data using the data loader
        for X, y in train_loader:
            # forward pass through the model
            y_pred = model(X)
            final_x = []

            # this grabs the last closing price from each batch and calculates the predicted and actual price change between that and the next day
            for batch in X:
                final_x.append(batch[-1].item())
            final_x = torch.tensor(final_x, dtype= torch.float)
            predicted_diff = (y_pred[:, 0] - final_x)
            actual_diff = (y[:, 0] - final_x)
            #print(actual_diff)

            # calculates MSE loss of predicted vs actual price change
            loss = loss_func(predicted_diff, actual_diff)

            # propogates the loss backwards through model
            optimizer.zero_grad()
            loss.backward()
            sum_loss += loss.item()
            optimizer.step()

        # Reports the training loss
        print("Average Training Loss for Epoch " + str(epoch + 1) + ":", sum_loss / len(x_train))

        # Puts the model into evaluation mode
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
        avg_val_err = sum_err / count

        scheduler.step()

        print("Validation Loss for Epoch " + str(epoch + 1) + ":", str(avg_val_err))
    if current:
        torch.save(model.state_dict(), "lstm_models\_" + ticker_name)
        joblib.dump(scaler, 'lstm_models\_' + ticker_name + '_scaler')

    return model, scaler

def rmse(predicted, actual):
    diff = predicted - actual
    mse = torch.mean(diff)**2
    return torch.sqrt(mse)

def evaluate_model(ticker):
    today_date = datetime.datetime.today()
    delta = datetime.timedelta(days= LOOK_BACK * 2)
    starting_date = today_date - delta
    start_data = str(starting_date).split()[0]

    stock_data = yf.download([ticker], start= start_data)
    close_prices = stock_data['Close'].values

    scaler = joblib.load('lstm_models\_' + ticker + '_scaler')
    test_data = scaler.fit_transform(close_prices.reshape(-1, 1))
    #test_data = close_prices.reshape(-1, 1)
    test_input = []

    for j in range(len(test_data) - LOOK_BACK):
        test_input.append(test_data[j : j + LOOK_BACK, :])
    #print(test_input)
    test_input = torch.tensor(np.array(test_input).reshape(-1, LOOK_BACK), dtype = torch.float)
    y_actual = torch.tensor(test_data[LOOK_BACK: ], dtype = torch.float)

    test_loader = data.DataLoader(data.TensorDataset(test_input, y_actual), batch_size= BATCH_SIZE)

    model = bi_lstm(LOOK_BACK, BATCH_SIZE, HIDDEN_SIZE, 1, NUM_LAYERS)
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

def predict_future_prices(ticker, load = True, _model = None, _scaler = None, fromDate = datetime.datetime.today()):
    sample_preds = []

    day_delta = datetime.timedelta(days= 1)
    delta = datetime.timedelta(days= LOOK_BACK * 2 + BATCH_SIZE)
    starting_date = fromDate - delta
    start_data = str(starting_date).split()[0]

    stock_data = yf.download([ticker], start= start_data, end= fromDate + day_delta)
    close_prices = stock_data['Close'].values
    final = close_prices[-1]

    if load:
        model = bi_lstm(LOOK_BACK, BATCH_SIZE, HIDDEN_SIZE, 1, NUM_LAYERS)
        model.load_state_dict(torch.load("lstm_models\_" + ticker))
        scaler = joblib.load('lstm_models\_' + ticker + '_scaler')
    elif _model != None and _scaler != None:
        model = _model
        scaler = _scaler
    model.eval()

    with torch.no_grad():
        counter = 0
        while counter < 3:

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


            sample_preds.append((np.mean(sampled_predictions) - final) / final)
            close_prices = np.append(close_prices, np.mean(sampled_predictions))
            model.dropout_off()

            counter += 1
    return sample_preds

#ticker_list = ['GOOG', 'PWR', 'VTI', 'MSFT', 'TSLA', 'NVDA']
#get_models(ticker_list)
#for t in ticker_list:
#    evaluate_model(t)
#print(predict_future_prices('GOOG', load = True, fromDate= datetime.date.fromisoformat('2023-07-18')))