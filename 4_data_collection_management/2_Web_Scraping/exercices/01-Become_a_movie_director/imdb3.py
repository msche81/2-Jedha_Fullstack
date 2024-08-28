import os 
import logging
import scrapy
from scrapy.crawler import CrawlerProcess

class imdb_spider(scrapy.Spider):
    # Name of your spider
    name = "imdb3"

    # Url to start your spider from 
    
    with open('01-Become_a_movie_director/url_list.txt') as f:
        start_urls = [line for line in f]

    # Callback function that will be called when starting your spider
    def parse(self, response):
            return {
                "cast": [actor.xpath('div[2]/a/text()').get() for actor in response.xpath('/html/body/div[2]/main/div/section[1]/div/section/div/div[1]/section[@data-testid="title-cast"]/div[2]/div[2]/div')]
                ,
                "storyline": response.xpath('/html/body/div[2]/main/div/section[1]/section/div[3]/section/section/div[3]/div[2]/div[1]/section/p/span[2]/text()').get()
                ,
                "genres": [genre.xpath('span/text()').get() for genre in response.xpath('/html/body/div[2]/main/div/section[1]/section/div[3]/section/section/div[3]/div[2]/div[1]/section/div[1]/div[2]/a')] 
                ,
                "title": response.xpath('/html/body/div[2]/main/div/section[1]/section/div[3]/section/section/div[2]/div[1]/h1/span/text()').get()
                ,
                "url": response.url.replace("https://www.imdb.com", '')
            }

# Name of the file where the results will be saved
filename = "imdb3.json"

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