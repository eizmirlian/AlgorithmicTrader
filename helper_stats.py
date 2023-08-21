import yfinance as yf
import datetime
import pandas as pd
import statistics
import math


def get_helper_stats(ticker, date = None, labeled =  False):
    if date == None:
        ten_day_hist = grab_ten_day_hist(ticker)
    else:
        ten_day_hist = grab_ten_day_hist(ticker, date = date)

    changes = get_perc_changes(ten_day_hist)
    momentum = compute_ten_day_momentum(ten_day_hist)
    volatility = compute_ten_day_volatility(ten_day_hist)
    special_volatility = compute_special_volatility(ten_day_hist)
    bollinger_high, bollinger_low = compute_bollinger_band(ten_day_hist)
    volume = return_volume(ten_day_hist)

    toReturn = [[momentum, volatility, special_volatility, bollinger_low, bollinger_high, volume]]
    if labeled and date != None:
        toReturn.append(label_datapoint(ticker, date))
    return toReturn

def grab_ten_day_hist(ticker, date = datetime.datetime.today()):

    
    delta = datetime.timedelta(days= 30)
    starting_date = date - delta

    day_delta = datetime.timedelta(days= 1)
    end_date = date + day_delta

    start_data = str(starting_date).split()[0]
    end_data = str(end_date).split()[0]
    stock_data = yf.download([ticker], start= start_data, end = end_data)
    num_days = stock_data.__len__()

    ten_day_hist = stock_data[num_days - 10:]
    return ten_day_hist

def get_perc_changes(ten_day_hist):
    closing_prices = ten_day_hist['Close'][-5:]
    prev = None
    changes = []
    for price in closing_prices:
        if prev == None:
            prev = price
        else:
            changes.append((price - prev) / prev * 100)
            prev = price
    return changes

def compute_ten_day_momentum(ten_day_hist):
    closing_prices = ten_day_hist['Close']
    return (closing_prices[-1] - closing_prices[0]) / closing_prices[0]

def compute_ten_day_volatility(ten_day_hist):
    percent_changes = []
    closing_prices = ten_day_hist['Close']
    for i in range(1, 10):
        percent_changes.append(closing_prices[i] / closing_prices[i - 1] - 1)
    return statistics.stdev(percent_changes)

def compute_special_volatility(ten_day_hist):
    percent_day_changes = []
    for i in range(1, 10):
        prev_close = ten_day_hist['Close'][i - 1]
        percent_day_changes.append((ten_day_hist['High'][i] / prev_close - 1) - (ten_day_hist['Low'][i] / prev_close - 1))
    return statistics.mean(percent_day_changes)

def compute_bollinger_band(ten_day_hist):
    closing_prices = ten_day_hist['Close']
    final = closing_prices[-1]
    ten_day_avg = statistics.mean(closing_prices)
    price_band = 1.5 * statistics.stdev(closing_prices)
    return [(ten_day_avg + price_band - final) / final, (ten_day_avg - price_band - final) / final]

def return_volume(ten_day_hist):
    return math.log10(ten_day_hist['Volume'][0])

def label_datapoint(ticker, date):
    day_delta = datetime.timedelta(days= 10)
    end_date = date + day_delta

    start_data = str(date).split()[0]
    end_data = str(end_date).split()[0]
    stock_data = yf.download([ticker], start= start_data, end = end_data)
    three_day_close = stock_data[:3]['Close'].values
    first, last = three_day_close[0], three_day_close[2]
    print(first, last)
    return first < last - (.01 * first)

#print(label_datapoint('MSFT', datetime.date.fromisoformat('2023-08-07')))
