from sklearn import preprocessing
from sklearn import svm
import numpy as np
import json
from svm_data_generation import load_svm_data, future_prediction_input
import datetime

def read_stored_datset(data, labeled = True):
    X = []
    y = []
    if labeled:
        for date in data.keys():
            stock_data = data[date]
            for stock in stock_data.keys():
                input_output = stock_data[stock]
                sample = input_output['Input']
                float_sample = []
                for s in sample:
                    float_sample.append(float(s))
                X.append(float_sample)
                label = input_output['Output']
                y.append(1 if label == 'Up' else 0)
    else:
        for stock in data.keys():
            input_output = data[stock]
            sample = input_output['Input']
            float_sample = []
            for s in sample:
                float_sample.append(float(s))
            X.append(float_sample)
    X = np.array(X)
    y = np.array(y)
    y.reshape(-1, 1)
    if labeled:
        return X, y
    else:
        return X

def retrieve_dataset():
    with open('svm_data.dat', 'r') as f:
        data = json.load(f)
        return read_stored_datset(data)
    
def generate_prediction_sample(tickers, past, dates):
    if past:
        data = load_svm_data(tickers, dates, load = False)
    else:
        data = future_prediction_input(tickers)
    return read_stored_datset(data, labeled = False)

def fit_svc(X, y):
    svc_scaler = preprocessing.StandardScaler()
    X_norm = svc_scaler.fit_transform(X)
    classifier = svm.SVC(C = 1.5, kernel= 'rbf', verbose= True, gamma = 1)
    classifier.fit(X_norm, y)
    return classifier, svc_scaler

def test_svc(ticker_list, dates):
    X, y = retrieve_dataset()
    X_test, y_test = generate_prediction_sample(ticker_list, True, dates)
    classifier, scaler = fit_svc(X, y)
    X_norm = scaler.transform(X_test)
    y_pred = classifier.predict(X_norm)
    num_correct = 0
    num_total = 0
    for i in range(y_test.size):
        print(y_test[i], y_pred[i])
        if y_test[i] == y_pred[i]:
            num_correct += 1
        num_total += 1
    accuracy = num_correct / num_total
    print(accuracy)

def predict_future(ticker_list):
    X_train, y_train = retrieve_dataset()
    X_in = generate_prediction_sample(ticker_list, False, None)
    classifier, scaler = fit_svc(X_train, y_train)
    X_norm = scaler.transform(X_in)
    y_pred = classifier.predict(X_norm)
    for i in range(y_pred.size):
        movement = 'up by at least 1%' if y_pred[i] == 1 else 'stay the same or down'
        print(ticker_list[i] + ' predicted to go ' + movement)




ticker_list = ['GOOG', 'MSFT', 'AMZN', 'META', 'COKE', 'NVDA', 'AMD']
dates = [datetime.date.fromisoformat('2023-08-01')]
predict_future(ticker_list)