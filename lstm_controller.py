import pandas as pd
import csv
import lstm
import datetime
import torch
from lstm_toolkit import bi_lstm

def scan_models(stock_names):
    toTrain = []
    toWrite = {}
    preexisting = stock_names.copy()
    with open("lstm_models\last_trained.csv", "r") as models_csv:
        reader = csv.DictReader(models_csv)
        today = datetime.date.today()
        for row in reader:
            ticker = row["STOCK_NAME"]
            training_date = datetime.date.fromisoformat(row["LAST_TRAINED"])
            week_delta = datetime.timedelta(days= 2)
            if ticker in stock_names:
                if training_date + week_delta <= today:
                    toTrain.append(ticker)
                    toWrite[ticker] = today.isoformat()
                else:
                    toWrite[ticker] = row["LAST_TRAINED"]
                preexisting.remove(ticker)
            else:
                toWrite[ticker] = row["LAST_TRAINED"]
        for stock in preexisting:
            toTrain.append(stock)
            toWrite[stock] = today.isoformat()
    if len(toTrain) >= 1:
        lstm.get_models(toTrain)
    with open("lstm_models\last_trained.csv", "w") as models_csv:
        writer = csv.DictWriter(models_csv, fieldnames= ["STOCK_NAME", "LAST_TRAINED"])
        writer.writeheader()
        for stock_name in toWrite.keys():
            writer.writerow({"STOCK_NAME" : stock_name, "LAST_TRAINED" : toWrite[stock_name]})

def evaluate_model(tickers):
    scan_models(tickers)
    for ticker in tickers:
        model = bi_lstm()
        model.load_state_dict(torch.load("lstm_models\_" + ticker))

def get_past_predictions(tickers, date):
    pred_dict = {}
    model_zoo = lstm.get_models(tickers, end_date= date)
    for ticker in tickers:
        pred_dict[ticker] = lstm.predict_future_prices(ticker, load= False, _model= model_zoo[ticker][0], _scaler= model_zoo[ticker][1], fromDate= date)
    #print(pred_dict)
    return pred_dict

def get_future_predictions(tickers):
    pred_dict = {}
    for ticker in tickers:
        pred_dict[ticker] = lstm.predict_future_prices(ticker)
    return pred_dict



#scan_models(["PWR", "GOOGL", "PW", "GTLB"])
            
    