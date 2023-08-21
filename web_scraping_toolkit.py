from bs4 import BeautifulSoup as BS
import requests as req
import datetime

def scrape_stock_news(ticker, date= datetime.date.today()):
    link = 'https://www.bloomberg.com/search?query=' + ticker
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'}
    page = req.get(link, headers= headers)
    scrape = BS(page.content, 'html.parser')

    headline_css_class = 'headline__3a97424275'
    summary_css_class = 'summary__a759320e4a'
    publishing_css_class = 'publishedAt__dc9dff8db4'


    news = {}

    week_delta = datetime.timedelta(days= 5)

    header_summary = []
    for headline in scrape.find_all('a', class_= headline_css_class):
        header_summary.append([headline.string.strip()])
    index = 0
    for summary in scrape.find_all('a', class_= summary_css_class):
        header_summary[index].append(summary.string.strip())
        index += 1
    index = 0
    for publishing_date in scrape.find_all('div', class_= publishing_css_class):
        published = datetime.datetime.strptime(publishing_date.string.strip(), "%B %d, %Y").date()
        if published > date - week_delta:
            news[published.isoformat()] = header_summary[index]
        index += 1
    print(news)
    return news

def scrape_market_news(link = 'https://www.schwab.com/learn/story/schwab-market-update', transcript = False):

    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'}
    page = req.get(link, headers= headers)
    scrape = BS(page.content, 'html.parser')

    summary_css_class = "bcn-marquee-story__summary bcn-ps-summary"
    intro_section_css_class = "col no-padding bcn-ps-body--l" if not transcript else "bcn-modal-content"
    date_css_class = 'bcn-ps-body--s'

    summary = scrape.find_all('div', class_= summary_css_class)[0].string.strip()

    publishing_date = scrape.find_all('span', class_= date_css_class)[1].text.strip().split('\n')[0]
    publishing_date = datetime.datetime.strptime(publishing_date, "%B %d, %Y").date().isoformat()
    

    first_paragraphs = []

    scraped_content = scrape.find_all('div', class_ = intro_section_css_class, limit= 2)

    intro_section = scraped_content[1] if not transcript else scraped_content[0]

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
        
    return summary, publishing_date, cleaned_text

def scrape_stock_schwab(tickers):
    links = ['https://www.schwab.com/learn/story/' + d + 'days-schwab-market-update-podcast#' for d in ['mon', 'tues', 'wednes', 'thurs', 'fri']]
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'}
    date_css_class = 'bcn-ps-body--s'
    content_css_class = "bcn-modal-content"
    all_relevant_lines = {}

    for link in links:
        relevant_lines = {}
        page = req.get(link, headers= headers)
        scrape = BS(page.content, 'html.parser')

        
        publishing_date = scrape.find_all('span', class_= date_css_class)[1].text.strip().split('\n')[0]
        publishing_date = datetime.datetime.strptime(publishing_date, "%B %d, %Y").date().isoformat()

        scraped_content = scrape.find_all(class_ = content_css_class)[0]

        paragraphs = []
        for paragraph in scraped_content.children:
            if paragraph != None:
                if paragraph.string == None:
                    for textChild in paragraph.children:
                        if textChild.string == None:
                            for listItem in textChild.children:
                                if listItem.string != None:
                                    paragraphs.append(listItem.string.strip())
                        else:
                            paragraphs.append(textChild.string.strip())
                else:
                    paragraphs.append(paragraph.string.strip())
        cleaned_text = []
        previous = ''
        for para in paragraphs:
            if len(para) == 0:
                continue
            if len(previous) < 50:
                previous += ' ' + para
            elif len(para) < 50:
                previous += ' ' + para
            elif previous.strip()[-1] != '.' and para.strip()[-1] == '.':
                previous += ' ' + para
            elif para.strip()[0] == '(':
                previous += ' ' + para
            else:
                cleaned_text.append(previous)
                previous = para
        for line in cleaned_text:
            open_paren_ind = -1
            for i in range(len(line)):
                if line[i] == '(':
                    open_paren_ind = i
                elif line[i] == ')' and open_paren_ind != -1:
                    if i - open_paren_ind < 6:
                        paren_content = line[open_paren_ind + 1 : i]
                        if paren_content in tickers:
                            if paren_content in relevant_lines.keys():
                                relevant_lines[paren_content].append(line)
                            else:
                                relevant_lines[paren_content] = [line]
                    open_paren_ind = -1
        all_relevant_lines[publishing_date] = relevant_lines
    return all_relevant_lines
