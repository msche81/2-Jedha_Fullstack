import os
import logging

import scrapy
from scrapy.crawler import CrawlerProcess

import json
file = open("02-Scraping_Yelp/spa-marseille.json")
file = json.load(file)
list_url = ["https://www.yelp.fr/" + element["url"] for element in file]

class YelpSpiderDetail(scrapy.Spider):
    # Name of your spider
    name = "yelp"

    # Starting URL
    start_urls = list_url

    # Parse function for form request
    def parse(self, response):
        name = response.xpath('/html/body/yelp-react-root/div[1]/div[3]/div[1]/div[1]/div/div/div[1]/h1/text()').get()
        address = response.xpath('/html/body/yelp-react-root/div[1]/div[5]/div/div[1]/div[2]/aside/section[1]/div/div[3]/div/div[1]/p[2]/text()').get()
        hours = response.xpath('/html/body/yelp-react-root/div[1]/div[4]/div/div/div[2]/div/div[1]/main/div[2]/section/div[2]/div[2]/div/div/table/tbody/tr/td[1]/ul/li/p/text()').getall()
        phone = response.xpath('/html/body/yelp-react-root/div[1]/div[5]/div/div[1]/div[2]/aside/section[1]/div/div[2]/div/div[1]/p[2]/text()').get()
        amenities = response.xpath('/html/body/yelp-react-root/div[1]/div[3]/div[1]/div[1]/div/div/span[3]/span/a/text()').getall()
        try:
          stars = response.xpath('/html/body/yelp-react-root/div[1]/div[3]/div[1]/div[1]/div/div/div[2]/div[1]/span/div').attrib["aria-label"] #response.css("div.i-stars__09f24__foihJ").attrib["aria-label"]
          n_votes = response.xpath('/html/body/yelp-react-root/div[1]/div[3]/div[1]/div[1]/div/div/div[2]/div[2]/span/a/text()').get()
        except KeyError:
          stars = None
          n_votes = None
        try:
          reviews = ["\n".join(review.xpath("div[1]/div[3]/p[@class='comment__09f24__D0cxf css-qgunke']/span/text()").getall()) for review in response.xpath("//section[@aria-label='Avis recommandés']/div[2]/ul/li")]
        except KeyError:
          reviews = None 

        return {
            "name":name,
            "url":response.url,
            "stars":stars,
            "n_votes":n_votes,
            "address":address,
            "phone":phone,
            "amenities":amenities,
            "reviews":reviews
        }

# Name of the file where the results will be saved
filename = "spa-marseille-detail" + ".json"

# If file already exists, delete it before crawling (because Scrapy will concatenate the last and new results otherwise)
if filename in os.listdir('02-Scraping_Yelp/'):
        os.remove('02-Scraping_Yelp/' + filename)

# Declare a new CrawlerProcess with some settings
process = CrawlerProcess(settings = {
    'USER_AGENT': 'Chrome/97.0',
    'LOG_LEVEL': logging.INFO,
    "AUTOTHROTTLE_ENABLED": True,
    "FEEDS": {
        '02-Scraping_Yelp/' + filename: {"format": "json"},
    }
})

# Start the crawling using the spider you defined above
process.crawl(YelpSpiderDetail)
process.start()