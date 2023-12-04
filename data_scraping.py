import os
from time import sleep

import pandas as pd
import requests
import scrapy
from bokeh.embed import file_html
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.resources import CDN
from bs4 import BeautifulSoup
from scrapy import Selector
from scrapy.crawler import CrawlerProcess

from selenium import webdriver
from selenium.webdriver.common.by import By

from selenium.webdriver.chrome.options import Options

import warnings
import re

from data_cleaning import Processor
from Charts import Charts


def get_wash_times_urls(urls):
    pages_visited = []
    page = 1

    while True:
        url = "https://www.washingtontimes.com/opinion/"
        print(page)
        print(len(urls))
        reqs = requests.get(url, params={"page": page})
        soup = BeautifulSoup(reqs.text, 'html.parser')

        for element in soup.find_all("h2", {"class": "article-headline"}):
            for link in element.find_all("a"):
                article = link.get('href')

                urls.append(f"http://www.washingtontimes.com{article}")
        page += 1
        if (len(urls) > 5000):
            break
    return urls


class WTSpider(scrapy.Spider):
    name = "tmp"

    def __init__(self, urls):
        super(WTSpider, self).__init__()
        self.urls = urls
        self.rows = []
        self.repattern = r"((Middle East|Israel|Gaza|Hamas|Palestinian Territories|Palestinian Authority|Syria|Turkey|Egypt|Iran|Saudi Arabia|Lebanon|Jordan|Qatar|Yemen|Palestine).*(Middle East|Israel|Gaza|Hamas|Palestinian Territories|Palestinian Authority|Syria|Turkey|Egypt|Iran|Saudi Arabia|Lebanon|Jordan|Qatar|Yemen|Palestine).*).*"

    def start_requests(self):
        count = 0
        for url in self.urls:
            count = count + 1
            yield scrapy.Request(url=url, callback=self.parse)
        print(count)

    def parse(self, response):
        tags = response.css(".block.whats-trending li *::text").getall()
        if any(x in tags for x in ["Middle East and north Africa", "Israel", "Gaza", "Hamas",
                                   "Palestinian Territories", "Palestinian Authority", "Syria",
                                   "Turkey", "Egypt", "Iran", "Saudi Arabia", "Lebanon", "Jordan",
                                   "Qatar", "Yemen"]):
            article_text = " ".join(response.css(".bigtext p::text").getall())
            author = response.css(".byline a::text").get()
            self.rows.append([article_text, tags, author, response.request.url])
        elif re.search(self.repattern,
                       " ".join(response.css(".bigtext p::text").getall())) is not None:
            article_text = " ".join(response.css(".bigtext p::text").getall())
            author = response.css(".byline a::text").get()
            self.rows.append([article_text, tags, author, response.request.url])

    def closed(self, reason):
        df = pd.DataFrame(self.rows, columns=['text', 'tags', 'author', 'url'])

        df.to_csv('./wash_times_me_op_eds.csv', index=False)


def get_wash_examiner_urls(urls):
    url = "https://www.washingtonexaminer.com/opinion"
    page = 1
    while True:
        reqs = requests.get(f"{url}/{page}")
        soup = BeautifulSoup(reqs.text, 'html.parser')

        print(soup)

        for element in soup.find_all("div", {"class": "SectionPromo-title"}):
            print("outer loop")
            for link in element.find_all("a"):
                print("made it")
                article = link.get('href')

                urls.append(article)
        page += 1
        if (len(urls) > 5):
            break
    return urls


def get_fox_urls():
    url = "https://www.foxnews.com/opinion"

    options = Options()
    options.add_argument('--headless=new')

    driver = webdriver.Chrome(options=options)
    driver.get(url)

    driver.implicitly_wait(15)

    count = 0

    try:
        while driver.find_element(By.CSS_SELECTOR, ".button.load-more") and count < 2000:
            try:
                driver.find_element(By.CSS_SELECTOR, ".button.load-more").click()
                count += 1
                print(count)
            except:
                print("sleeping")
                print(
                    len([x for x in Selector(text=driver.page_source).xpath('//h4[@class="title"]/a/@href').getall() if
                         "/opinion/" in x]))
                sleep(1)
    except:
        print("might have been sent off somewhere")

    response = Selector(text=driver.page_source)

    f = open("page_source.txt", "w")
    f.write(driver.page_source)
    f.close()

    return [f"https://www.foxnews.com{x}" for x in response.xpath('//h4[@class="title"]/a/@href').getall() if
            "/opinion/" in x]


def get_guardian_urls():
    url = "https://www.theguardian.com/commentisfree/all"

    urls = []
    pages_visited = []

    while True:

        reqs = requests.get(url)
        pages_visited.append(url)
        soup = BeautifulSoup(reqs.text, 'html.parser')

        for element in soup.find_all("div", {"class": "u-cf index-page"}):
            for link in element.find_all("a"):
                article = link.get('href')
                if re.match(".+/all", article):
                    continue
                if re.match(".+?page=\d", article):
                    m = re.search(r'\d+$', article)
                    url = article
                elif "commentisfree" in article and article not in urls:
                    urls.append(article)
        if (url in pages_visited) or (len(pages_visited) > 500):
            return urls


class FoxSpider(scrapy.Spider):
    name = "fox"

    custom_settings = {
        'DOWNLOAD_DELAY': 1
    }

    def __init__(self, urls):
        super(FoxSpider, self).__init__()
        self.urls = urls
        self.rows = []
        self.repattern = r"((Middle East|Israel|Gaza|Hamas|Palestinian Territories|Palestinian Authority|Syria|Turkey|Egypt|Iran|Saudi Arabia|Lebanon|Jordan|Qatar|Yemen|Palestine).*(Middle East|Israel|Gaza|Hamas|Palestinian Territories|Palestinian Authority|Syria|Turkey|Egypt|Iran|Saudi Arabia|Lebanon|Jordan|Qatar|Yemen|Palestine).*).*"

    def start_requests(self):
        for url in self.urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        if re.search(self.repattern,
                     " ".join(response.css(".article-body p::text").getall())) is not None:
            article_text = " ".join(response.css(".article-body p::text").getall())
            author = response.css(".author-byline a::text").get()
            self.rows.append([article_text, author, response.request.url])
        if len(self.rows) >= 250:
            return

    def closed(self, reason):
        df = pd.DataFrame(self.rows, columns=['text', 'author', 'url'])

        df.to_csv('./fox_me_op_eds.csv', index=False)


class GuardianSpider(scrapy.Spider):
    name = "guardian"

    custom_settings = {
        'DOWNLOAD_DELAY': 1
    }

    def __init__(self, urls):
        super(GuardianSpider, self).__init__()
        self.urls = urls
        self.rows = []
        self.repattern = r"((Middle East|Israel|Gaza|Hamas|Palestinian Territories|Palestinian Authority|Syria|Turkey|Egypt|Iran|Saudi Arabia|Lebanon|Jordan|Qatar|Yemen|Palestine|West Bank|Syria).*(Middle East|Israel|Gaza|Hamas|Palestinian Territories|Palestinian Authority|Syria|Turkey|Egypt|Iran|Saudi Arabia|Lebanon|Jordan|Qatar|Yemen|Palestine|West Bank|Syria).*).*"

    def start_requests(self):
        for url in self.urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        tags = response.css(".dcr-1apjr72 a::text").getall()

        if re.search(self.repattern,
                     " ".join(response.css(".article-body p::text").getall())) is not None:
            article_text = " ".join(response.css("[id=\"maincontent\"]::text").getall())
            author = response.css("[rel=\"author\"]::text").get()
            self.rows.append([article_text, author, response.request.url])

        elif any(x in tags for x in ["Middle East and north Africa", "Israel-Hamas war",
                                     "Israel", "Gaza", "Hamas", "Palestinian Territories",
                                     "Palestinian Authority", "Palestine", "Syria", "Turkey",
                                     "Egypt", "Iran", "Saudi Arabia", "Lebanon", "Jordan",
                                     "Qatar", "Yemen", "West Bank", "Syria"]):
            article_text = " ".join(response.css(".dcr-ty818o p::text").getall())
            author = response.css("[rel=\"author\"]::text").get()
            self.rows.append([article_text, tags, author, response.request.url])

        if len(self.rows) >= 250:
            self.closed("manual")

    def closed(self, reason):
        df = pd.DataFrame(self.rows, columns=['text', 'tags', 'author', 'url'])

        df.to_csv('./guardian_me_op_eds.csv', index=False)


def fox_process():
    urls = get_fox_urls()
    print(len(urls))
    process = CrawlerProcess()

    process.crawl(FoxSpider, urls=urls)
    process.start()


def guardian_process():
    urls = get_guardian_urls()
    print(len(urls))
    # urls = ["https://www.theguardian.com/commentisfree/2023/oct/16/why-i-believe-the-bds-movement-has-never-been-more-important-than-now"]
    process = CrawlerProcess()

    process.crawl(GuardianSpider, urls=urls)
    process.start()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    if not os.path.isfile("fox_me_op_eds.csv"):
        fox_process()

    df_fox = pd.read_csv("fox_me_op_eds.csv")

    if not os.path.isfile("guardian_me_op_eds.csv"):
        guardian_process()

    df_guardian = pd.read_csv("guardian_me_op_eds.csv").drop("tags", axis=1)

    if not os.path.isfile("cleaned_data.csv"):
        df = pd.concat([df_fox, df_guardian], ignore_index=True).drop_duplicates()
        df = Processor.clean_df(df)
        df.to_csv("./cleaned_data.csv")

    df = pd.read_csv("cleaned_data.csv", converters={'spacy_tokenized_article': pd.eval,
                                                     'word_tokenized_article': pd.eval,
                                                     "article_without_keywords": pd.eval,
                                                     'extreme_tokens': pd.eval})

    tfidf = Processor.process_df(df)

    predicted = Processor.run_clustering(tfidf, df)
    predicted.to_csv("./predicted.csv")

    #Charts().mean_polarity_per_cluster_graph(predicted, "dbscan")
    # Charts().mean_polarity_per_cluster_graph(predicted, "kmeans")
    # Charts().mean_polarity_per_cluster_graph(predicted, "dbscan", "spacy_tokenized_article")
    # Charts().mean_polarity_per_cluster_graph(predicted, "kmeans", "spacy_tokenized_article")

    km_polarity_cluster = Charts().bokeh_mean_polarity_per_cluster_graph(predicted, "kmeans")
    dbscan_polarity_cluster = Charts().bokeh_mean_polarity_per_cluster_graph(predicted, "dbscan")

    layout = row(km_polarity_cluster, dbscan_polarity_cluster)
    curdoc().theme = 'dark_minimal'

    html = file_html(layout, CDN, "Layout")

    with open("tmp.html", "w") as f:
        f.write(html)
        f.close()


    # Processor.wordcloud_by_most_frequent_keyword(predicted, "article_without_keywords")

    for x in ["kmeans", "dbscan"]:
        for y in ["article_without_keywords", "extreme_tokens", "spacy_tokenized_article", "word_tokenized_article"]:

            Processor.most_frequent_keyword_wordcloud_by_cluster(predicted, x, y)

    # Processor.most_frequent_keyword_wordcloud_by_source(df, "Guardian", "extreme_tokens")
    #Processor.most_frequent_keyword_wordcloud_by_source(df, "Fox", "extreme_tokens")
    # Processor.most_frequent_keyword_wordcloud_by_source(df, "Fox")
    for y in ["article_without_keywords", "extreme_tokens", "spacy_tokenized_article", "word_tokenized_article"]:
        chord_diagram = Processor.create_source_chord_diagram(df, y)
        html = file_html(chord_diagram, CDN, "Layout", theme="dark_minimal")

        with open(f"{y}.html", "w") as f:
            f.write(html)
            f.close()

   #Charts().network_graph(df)
