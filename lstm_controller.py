import pandas as pd
import csv
import lstm
import datetime
import tensorflow as tf
from tensorflow import keras
from keras import layers

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
            week_delta = datetime.timedelta(days= 7)
            if ticker in stock_names:
                if training_date + week_delta <= today:
                    toTrain.append(ticker)
                    toWrite[ticker] = today.isoformat()
                else:
                    toWrite[ticker] = row["LAST_TRAINED"]
                preexisting.remove(ticker)
        for stock in preexisting:
            toTrain.append(stock)
            toWrite[stock] = today.isoformat()
    with open("lstm_models\last_trained.csv", "w") as models_csv:
        writer = csv.DictWriter(models_csv, fieldnames= ["STOCK_NAME", "LAST_TRAINED"])
        writer.writeheader()
        for stock_name in toWrite.keys():
            writer.writerow({"STOCK_NAME" : stock_name, "LAST_TRAINED" : toWrite[stock_name]})
        lstm.get_data(toTrain)

def evaluate_model(tickers):
    scan_models(tickers)
    for ticker in tickers:
        model = keras.Model.load_weights("lstm_models\_" + ticker)

scan_models(["PWR", "GOOGL", "PW", "GTLB"])
            
    