"""

Script retrieves news article URLs from Il Giornale sitemap and scrapes all news articles from last 5 years

Il Giornale sitemap index found from robots.txt: https://www.ilgiornale.it/robots.txt

"""

# Import libraries
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from newspaper import Article
from date_guesser import guess_date, Accuracy
from time import sleep
from random import randint
import os

# Constants
SITEMAP_INDEX_URL = "https://www.ilgiornale.it/sitemap/indice.xml"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0",
}

# Path
CURRENT_DIRECTORY = os.getcwd()

# Dates
current_date = datetime.now()
current_year = current_date.year
current_month = current_date.month
current_day = current_date.day


# Get xml files from DeSpeld sitemap
def get_sitemap_files(soup):
    """Function retrieves XML files from Il Giornale sitemap
    Args:
        soup (object): soup object of scraped website
    Returns:
        found_sitemap_files (list): list of retrieved sitemap XML paths
    """

    found_sitemap_files = set()

    for sitemap in soup.findAll("sitemap"):
        for loc in sitemap.findAll("loc"):
            link = loc.text
            match = re.search(r"\d{4}/\d{2}.xml", link)
            if link and "/sitemap/" in link and match:
                found_sitemap_files.add(link)

    return list(found_sitemap_files)


# Get news article URLs from sitemap XMLs
def get_news_article_urls(xml_path: str):
    """Function retrieves news article URLs from Il Giornale sitemap XMLs
    Args:
        xml_path (str): link of XML file where news article URLs are contained
    Returns:
        found_links (list): found news article URLs
    """

    page = requests.get(xml_path, timeout=50)
    soup = BeautifulSoup(page.content, "xml")

    found_news_article_links = set()

    for sitemap in soup.findAll("url"):
        for loc in sitemap.findAll("loc"):
            link = loc.text

            # Only get links that have /yyyy/mm/dd/ structure in them
            if "/news/" in link:
                found_news_article_links.add(link)

    return list(found_news_article_links)


# Clean html from text
def clean_html(text: str):
    """Function takes text and removes html code from it
    Args:
        text (str): text
    Returns:
       text (str): text without HTML code
    """

    re_remove_hmtl = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    text = re.sub(re_remove_hmtl, "", str(text))

    return text


# Scrape news article data
def get_news_article_data(article_url: str):
    """Function scrapes article URL and retrieves article data
    Args:
        article_url (str): URL of the news article
    Returns:
        article_data_dict (dict): article_data
    """

    page = None
    title = None
    text = None
    document = None
    article_dict = {}

    # Random sleep before initiating article scraping
    sleep(randint(0, 3))

    try:
        article = Article(url=article_url)
        article.download()
        article.parse()
    except Exception as e:
        print(e)
        article = None

    if article:
        title = article.title
        title = clean_html(text=title).strip()
        text = article.text

    # Get text from Document and clean html
    article_dict["article_url"] = article_url
    article_dict["article_title"] = title
    article_dict["article_text"] = text

    # Try to guess date from response
    try:

        if page:

            guess = guess_date(url=article_url)
            article_dict["publish_date"] = guess.date
            article_dict["published_method_found"] = guess.method

            if guess.accuracy is Accuracy.PARTIAL:
                article_dict["published_guess_accuracy"] = "partial"
            if guess.accuracy is Accuracy.DATE:
                article_dict["published_guess_accuracy"] = "date"
            if guess.accuracy is Accuracy.DATETIME:
                article_dict["published_guess_accuracy"] = "datetime"
            if guess.accuracy is Accuracy.NONE:
                article_dict["published_guess_accuracy"] = None

        else:
            article_dict["publish_date"] = None
            article_dict["published_method_found"] = None
            article_dict["published_guess_accuracy"] = None

    except:
        article_dict["publish_date"] = None
        article_dict["published_method_found"] = None
        article_dict["published_guess_accuracy"] = None

    if article_dict["publish_date"]:
        article_dict["publish_date"] = article_dict["publish_date"].strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    return article_dict


if __name__ == "__main__":

    # Make request to De Speld sitemap
    page = requests.get(SITEMAP_INDEX_URL, timeout=50, headers=HEADERS)
    soup = BeautifulSoup(page.content, "xml")

    # Get sitemap XML paths
    found_sitemap_files = get_sitemap_files(soup=soup)

    file_num = 0

    for sitemap_path in found_sitemap_files:

        # Get news article URLs
        found_news_article_links = get_news_article_urls(xml_path=sitemap_path)

        # Get article data
        article_data = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            executor_object = executor.map(
                get_news_article_data, found_news_article_links
            )
            for found_news_article_data in executor_object:
                if found_news_article_data:
                    article_data.append(found_news_article_data)

        # Transform month and date values
        if current_month < 10:
            current_month_str = f"0{current_month}"
        else:
            current_month_str = str(current_month)

        if current_day < 10:
            current_day_str = f"0{current_day}"
        else:
            current_day_str = str(current_day)

        # Store retrieved data to a dataframe
        df = pd.DataFrame.from_dict(article_data, orient="columns")
        if len(df) > 1:
            df.to_csv(
                f"./{CURRENT_DIRECTORY}/giornale/il_giornale_scraped_articles_{current_year}{current_month_str}{current_day_str}_filenumber_{file_num}.csv",
                index=False,
            )
            file_num += 1
