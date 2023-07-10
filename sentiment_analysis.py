from bs4 import BeautifulSoup as BS
import requests as req
import openai
import datetime

GPT_API_KEY = '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'

def generate_stock_sentiment_scores(ticker):
    news = scrape_stock_news(ticker, 3)
    scores = []
    for article in news:
        new_score = gpt_ticker_sentiment_analysis(article[0], article[1], ticker)
        if new_score != 0:
            scores.append(new_score)

def generate_market_sentiment_scores():
    summary, paragraphs = scrape_market_news()
    scores = []
    for para in paragraphs:
        scores.append(gpt_market_sentiment_analysis(summary, para))
    print(scores)

def scrape_stock_news(ticker, num_headlines):
    link = 'https://www.bloomberg.com/search?query=' + ticker
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'}
    page = req.get(link, headers= headers)
    scrape = BS(page.content, 'html.parser')

    headline_css_class = 'headline__3a97424275'
    summary_css_class = 'summary__a759320e4a'


    news = []

    week_delta = datetime.timedelta(days= 14)
    today = datetime.date.today()

    header_summary = []
    for headline in scrape.find_all('a', class_= headline_css_class,  limit = num_headlines):
        header_summary.append([headline.string.strip()])
    index = 0
    for summary in scrape.find_all('a', class_= summary_css_class,  limit = num_headlines):
        header_summary[index].append(summary.string.strip())
        index += 1
    index = 0
    for publishing_date in scrape.find_all('div', class_= 'publishedAt__dc9dff8db4',  limit = num_headlines):
        published = datetime.datetime.strptime(publishing_date.string.strip(), "%B %d, %Y").date()
        if published > today - week_delta:
            news.append(header_summary[index])
        index += 1
    print(news)
    return news

def scrape_market_news():
    link = 'https://www.schwab.com/learn/story/schwab-market-update'
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'}
    page = req.get(link, headers= headers)
    scrape = BS(page.content, 'html.parser')

    summary_css_class = "bcn-marquee-story__summary bcn-ps-summary"
    intro_section_css_class = "col no-padding bcn-ps-body--l"

    summary = scrape.find_all('div', class_= summary_css_class)[0].string.strip()

    first_paragraphs = []

    intro_section = scrape.find_all('div', class_ = intro_section_css_class, limit= 2)[1]
    for paragraph in intro_section.children:
        if paragraph != None:
            if paragraph.string == None:
                for textChild in paragraph.children:
                    if textChild.string != None:
                        first_paragraphs.append(textChild.string.strip())
            else:
                first_paragraphs.append(paragraph.string.strip())
    cleaned_text = []
    curr_paragraph = None
    for text in first_paragraphs:
        if curr_paragraph == None:
            curr_paragraph = text + ' '
        elif len(text) < 100:
            curr_paragraph += text + ' '
        else:
            curr_paragraph += text
            if len(curr_paragraph) > 500:
                cleaned_text.append(curr_paragraph)
                curr_paragraph = None

    if len(curr_paragraph) > 150 and len(cleaned_text) < 3:
        cleaned_text.append(curr_paragraph)
    else:
        cleaned_text[-1] += curr_paragraph
    if len(cleaned_text) > 3:
        cleaned_text = cleaned_text[: 3]
    for i in range(len(cleaned_text)):
        cleaned_text[i] = cleaned_text[i].strip()
        
    return summary, cleaned_text



def gpt_ticker_sentiment_analysis(headline, summary, ticker):
    openai.api_key = GPT_API_KEY
    response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {'role': 'user', 'content': 'You will work as a Sentiment Analysis for Financial News. I will share a stock name, a news headline, and a summary, and you will respond with a score of how bullish or bearish it is. Your response should be only the score. The range is -10 for the most bearish to 10 for the most bullish. If the headline and summary are unrelated to the stock, you will report a neutral score of 0. No further explanation. Got it?'},
        {'role': 'system', 'content': 'Got it! Please provide me with the stock name, news headline, and summary, and I will provide you with a sentiment score ranging from -10 (most bearish) to 10 (most bullish).'},
        {'role': 'user', 'content': 'Stock: ' + ticker + '\n' + 'Headline: ' + headline + '\n' + 'Summary: ' + summary}], temperature= 0,)
    
    score = response['choices'][0]['message']['content']
    return int(score)


def gpt_market_sentiment_analysis(summary, paragraph):
    openai.api_key = GPT_API_KEY
    response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {'role': 'user', 'content': 'You will work as a Sentiment Analysis for Financial News. I will share a summary and a paragraph from an article about the current state of the market. You will respond with a score of how bullish or bearish the content of the paragraph is using the context of the summary. Your response should be only the score. The range is -10 for the most bearish to 10 for the most bullish. Got it?'},
        {'role': 'system', 'content': 'Got it! Please go ahead and provide me with the summary and the paragraph from the article, and I will provide you with a score indicating the bullishness or bearishness of the content.'},
        {'role': 'user', 'content': 'Summary: ' + summary + '\n' + 'Paragraph: ' + paragraph}], temperature= 0,)
    
    score = response['choices'][0]['message']['content']
    return int(score)
