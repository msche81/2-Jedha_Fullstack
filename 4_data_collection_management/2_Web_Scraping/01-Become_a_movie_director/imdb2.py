import os
import logging
import scrapy
from scrapy.crawler import CrawlerProcess

class imdb_spider(scrapy.Spider):
    # Name of your spider
    name = "imdb2"

    # Url to start your spider from 
    start_urls = [
        'https://www.imdb.com/chart/boxoffice',
    ]

    # Callback function that will be called when starting your spider
    def parse(self, response):
        movies = response.xpath('/html/body/div[2]/main/div/div[3]/section/div/div[2]/div/ul/li')
        for movie in movies:
            yield {
                "ranking": movie.xpath('div[2]/div/div/div/a/h3/text()').get().split(".")[0]
                ,
                "title": movie.xpath('div[2]/div/div/div/a/h3/text()').get().split(".")[1].strip(" ")
                , 
                "url": movie.xpath('div[2]/div/div/div/a').attrib["href"]
                ,
                "total_earnings": movie.xpath('div[2]/div/div/ul/li[2]/span[2]/text()').get()
                ,
                "rating": movie.xpath('div[2]/div/div/span/div/span/text()').get()
                ,
                "nb_voters": movie.xpath('div[2]/div/div/span/div/span/span/text()').getall()[1]
                }

# Name of the file where the results will be saved
filename = "imdb2.json"

# If file already exists, delete it before crawling (because Scrapy will 
# concatenate the last and new results otherwise)
if filename in os.listdir('01-Become_a_movie_director/'):
        os.remove('01-Become_a_movie_director/' + filename)

# Declare a new CrawlerProcess with some settings
## USER_AGENT => Simulates a browser on an OS
## LOG_LEVEL => Minimal Level of Log 
## FEEDS => Where the file will be stored 
## More info on built-in settings => https://docs.scrapy.org/en/latest/topics/settings.html?highlight=settings#settings
process = CrawlerProcess(settings = {
    'USER_AGENT': 'Chrome/97.0',
    'LOG_LEVEL': logging.INFO,
    "FEEDS": {
        '01-Become_a_movie_director/' + filename : {"format": "json"},
    }
})

# Start the crawling using the spider you defined above
process.crawl(imdb_spider)
process.start()