# Import os => Library used to easily manipulate operating systems
## More info => https://docs.python.org/3/library/os.html
import os 

# Import logging => Library used for logs manipulation 
## More info => https://docs.python.org/3/library/logging.html
import logging

# Import scrapy and scrapy.crawler 
import scrapy
from scrapy.crawler import CrawlerProcess

class QuotesSpider(scrapy.Spider):

    # Name of your spider
    name = "quotes"

    # Url to start your spider from 
    start_urls = [
        'http://quotes.toscrape.com/page/1/',
    ]

    # Callback function that will be called when starting your spider
    # It will get text, author and tags of all the <div> with class="quote"
    def parse(self, response):
        n = 10
        for i in range(n):
            i = i + 1
            yield {
                'text': response.xpath('/html/body/div/div[2]/div[1]/div[{}]/span[1]/text()'.format(i)).get(),
                'author': response.xpath('/html/body/div/div[2]/div[1]/div[{}]/span[2]/small/text()'.format(i)).get(),
                'tags': response.xpath('/html/body/div/div[2]/div[1]/div[{}]/div/a/text()'.format(i)).getall(),
            }
            
# Name of the file where the results will be saved
filename = "2_quotes.json"

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
process.crawl(QuotesSpider)
process.start()