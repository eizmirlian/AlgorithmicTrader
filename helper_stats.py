import yfinance as yf
import datetime
import pandas as pd
import statistics


def get_helper_stats(ticker):
    ten_day_hist = grab_ten_day_hist(ticker)
    momentum = compute_ten_day_momentum(ten_day_hist)
    volatility = compute_ten_day_volatility(ten_day_hist)
    special_volatility = compute_special_volatility(ten_day_hist)
    bollinger_high, bollinger_low = compute_bollinger_band(ten_day_hist)
    volume = return_volume(ten_day_hist)
    return [momentum, volatility, special_volatility, bollinger_low, bollinger_high, volume]

def grab_ten_day_hist(ticker):

    today_date = datetime.datetime.today()
    delta = datetime.timedelta(days= 30)
    starting_date = today_date - delta

    start_data = str(starting_date).split()[0]
    stock_data = yf.download([ticker], start= start_data)
    num_days = stock_data.__len__()

    ten_day_hist = stock_data[num_days - 10:]
    return ten_day_hist

def compute_ten_day_momentum(ten_day_hist):
    closing_prices = ten_day_hist['Close']
    return closing_prices[-1] - closing_prices[0]

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
    ten_day_avg = statistics.mean(closing_prices)
    price_band = 1.5 * statistics.stdev(closing_prices)
    return [ten_day_avg + price_band, ten_day_avg - price_band]

def return_volume(ten_day_hist):
    return ten_day_hist['Volume'][0]


print(get_helper_stats('msft'))
