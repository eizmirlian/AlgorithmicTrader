import lstm_controller
import sentiment_analysis
import helper_stats
import sentiment_analysis
import json
import time
import statistics
import datetime

def load_svm_data(tickers, dates, load = True):
    toLoad = {}
    for date in dates:
        date_str = date.isoformat()
        toLoad[date_str] = assemble_dataset(tickers, date)
    if load:
        load_json('svm_data.dat', toLoad, override= True)
    else:
        return toLoad


def assemble_dataset(tickers, date):
    data = {}
    date_str = date.isoformat()
    predictions = lstm_controller.get_past_predictions(tickers, date)
    queried_scores = retrieve_stock_sent_scores(tickers, date)
    with open('market_sent_score_history.dat','r') as f:
        scores = json.load(f)
        stored_keys = scores.keys()
        if date_str in stored_keys:
            market_sent_score = scores[date_str]
        else:
            return 'No market sentiment score available for this date'
    for ticker in tickers:
        future_pred = predictions[ticker]
        data[ticker] = {'Input':[], 'Output': ''}
        add_stats, label = helper_stats.get_helper_stats(ticker, date= date, labeled= True)

        data[ticker]['Output'] = 'Up' if label else 'Down'

        for day in future_pred:
            data[ticker]['Input'].append(str(day))
        data[ticker]['Input'].append(str(queried_scores[ticker]))
        data[ticker]['Input'].append(str(market_sent_score))
        
        for stat in add_stats:
            data[ticker]['Input'].append(str(stat))
    return data
               
def future_prediction_input(tickers):
    today = datetime.date.today()
    week_delta = datetime.timedelta(days = 7)
    date_str = today.isoformat()
    data = {}
    lstm_controller.scan_models(tickers)
    predictions = lstm_controller.get_future_predictions(tickers)
    score_needed = False
    with open('market_sent_score_history.dat','r') as f:
        scores = json.load(f)
        stored_keys = scores.keys()
        if date_str in stored_keys:
            market_sent_score = scores[date_str]
        else:
            score_needed = True
    if score_needed:
        market_sent_score = collect_market_sentiment_scores(sameDay = True)[date_str]
    score_needed = False
    with open('stock_sent_score_history.dat','r') as f:
        scores = json.load(f)
        stored_keys = scores.keys()
        if date_str not in stored_keys:
            score_needed = True
    if score_needed:
        collect_stock_sentiment_scores(tickers, today - week_delta)
    queried_scores = retrieve_stock_sent_scores(tickers, today)

    for ticker in tickers:
        future_pred = predictions[ticker]
        data[ticker] = {'Input':[]}
        add_stats = helper_stats.get_helper_stats(ticker)[0]

        for day in future_pred:
            data[ticker]['Input'].append(str(day))
        data[ticker]['Input'].append(str(queried_scores[ticker]))
        data[ticker]['Input'].append(str(market_sent_score))
        
        for stat in add_stats:
            data[ticker]['Input'].append(str(stat))
    print(data)
    return data
    
    
        
def collect_market_sentiment_scores(sameDay = False):
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']
    scraped_sent_scores = {}
    if sameDay:
        scores, date = sentiment_analysis.generate_market_sentiment_scores()
        avg_score = statistics.average(scores)
        scraped_sent_scores[date] = avg_score
        load_json('market_sent_score_history.dat', scraped_sent_scores)
        time.sleep(60)
        return scraped_sent_scores
    else:
        for d in days:
            market_update_link = 'https://www.schwab.com/learn/story/' + d + 's-schwab-market-update-podcast#'
            scores, date = sentiment_analysis.generate_market_sentiment_scores(link = market_update_link)
            if date not in scraped_sent_scores.keys():
                avg_score = statistics.average(scores)
                scraped_sent_scores[date] = avg_score
            time.sleep(60)
        load_json('market_sent_score_history.dat', scraped_sent_scores)
        

def collect_stock_sentiment_scores(tickers, start):
    data = sentiment_analysis.generate_stock_supplement_scores(tickers)
    time.sleep(60)
    for ticker in tickers:
        date_scores = sentiment_analysis.generate_stock_sentiment_scores(ticker, start)
        for publishing_date in date_scores.keys():
            if publishing_date in data.keys():
                if ticker in data[publishing_date].keys():
                    old_score = data[publishing_date][ticker]
                    if old_score == 0:
                        data[publishing_date][ticker] = date_scores[publishing_date]
                    else:
                        data[publishing_date][ticker] = (date_scores[publishing_date] + old_score) / 2
                else:
                    data[publishing_date][ticker] = date_scores[publishing_date]
            else:
                data[publishing_date] = {ticker : date_scores[publishing_date]}
    load_json('stock_sent_score_history.dat', data)

def retrieve_stock_sent_scores(tickers, date):
    with open('stock_sent_score_history.dat','r') as f:
        data = json.load(f)
        day_delta = datetime.timedelta(days = 3)
        toReturn = {}
        for publishing_date in data.keys():
            publishing_obj = datetime.date.fromisoformat(publishing_date)
            if date - day_delta <= publishing_obj and date > publishing_obj:
                stored_scores = data[publishing_date]
                for ticker in tickers:
                    if ticker in stored_scores.keys():
                        if ticker in toReturn.keys():
                            toReturn[ticker].append(stored_scores[ticker])
                        else:
                            toReturn[ticker] = [stored_scores[ticker]]
    for ticker in tickers:
        if ticker in toReturn.keys():
            if len(toReturn[ticker]) >= 1:
                toReturn[ticker] = statistics.mean(toReturn[ticker])
        else:
            toReturn[ticker] = 0
    return toReturn

def load_json(filename, toLoad, override= False):
    with open(filename,'r') as f:
        data = json.load(f)
        stored_keys = data.keys()
        new_keys = toLoad.keys()
        for d in new_keys:
            if override or d not in stored_keys:
                data[d] = toLoad[d]
    with open(filename, 'w') as f:
        json.dump(data, f, sort_keys= True)

DATA_GEN_MODE = False
MARKET_COLLECTION = False
STOCK_COLLECTION = False

dates_str = ['2023-08-11', '2023-08-14', '2023-08-15', '2023-08-16', '2023-08-17']
dates = [datetime.date.fromisoformat(d) for d in dates_str]
ticker_list = ['GOOG', 'MSFT', 'PWR', 'META', 'AMZN', 'AAPL', 'TSLA', 'T', 'AMD', 'COKE', 'DIS', 'NVDA', 'INTC', 'CVX', 'XOM', 'F', 'ROKU', 'PG', 'RUN']

if MARKET_COLLECTION:
    collect_market_sentiment_scores()
if STOCK_COLLECTION:
    collect_stock_sentiment_scores(ticker_list, dates[0])
if DATA_GEN_MODE:
    load_svm_data(ticker_list, dates)