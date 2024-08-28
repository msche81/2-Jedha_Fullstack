# Import os => Library used to easily manipulate operating systems
## More info => https://docs.python.org/3/library/os.html
import os 

# Import logging => Library used for logs manipulation 
## More info => https://docs.python.org/3/library/logging.html
import logging

# Import scrapy and scrapy.crawler 
import scrapy
from scrapy.crawler import CrawlerProcess

class RandomQuoteSpider(scrapy.Spider):
    # Name of your spider
    name = "randomquote"

    # Url to start your spider from 
    start_urls = [
        'http://quotes.toscrape.com/random',
    ]

    # Callback function that will be called when starting your spider
    # It will get text, author and tags of the first <div> with class="quote"
    def parse(self, response):
        return {
            'text': response.xpath("/html/body/div/div[2]/div[1]/div/span[1]/text()").get(),
            'author': response.xpath('/html/body/div/div[2]/div[1]/div/span[2]/small/text()').get(),
            'tags': response.xpath('/html/body/div/div[2]/div[1]/div/div/a/text()').getall(),
        }

# Name of the file where the results will be saved
filename = "1_randomquote.json"

# If file already exists, delete it before crawling (because Scrapy will 
# concatenate the last and new results otherwise)
if filename in os.listdir('src/'):
        os.remove('src/' + filename)

# Declare a new CrawlerProcess with some settings
## USER_AGENT => Simulates a browser on an OS
## LOG_LEVEL => Minimal Level of Log 
## FEEDS => Where the file will be stored 
## More info on built-in settings => https://docs.scrapy.org/en/latest/topics/settings.html?highlight=settings#settings
process = CrawlerProcess(settings = {
    'USER_AGENT': 'Chrome/97.0',
    'LOG_LEVEL': logging.INFO,
    "FEEDS": {
        'src/' + filename : {"format": "json"},
    }
})

# Start the crawling using the spider you defined above
process.crawl(RandomQuoteSpider)
process.start()