# Import os => Library used to easily manipulate operating systems
## More info => https://docs.python.org/3/library/os.html
import os 

# Import logging => Library used for logs manipulation 
## More info => https://docs.python.org/3/library/logging.html
import logging

# Import scrapy and scrapy.crawler 
import scrapy
from scrapy.crawler import CrawlerProcess

class QuotesMultipleSpider(scrapy.Spider):

    # Name of your spider
    name = "quotesmultiplepages"

    # Url to start your spider from 
    start_urls = [
        'http://quotes.toscrape.com/page/1/',
    ]

    # Callback function that will be called when starting your spider
    # It will get text, author and tags of the <div> with class="quote"
    # /html/body/div/div[2]/div[1]/div[1]/span[1]
    def parse(self, response):
        quotes = response.xpath('/html/body/div/div[2]/div[1]/div')
        for quote in quotes:
            yield {
                'text': quote.xpath('span[1]/text()').get(),
                'author': quote.xpath('span[2]/small/text()').get(),
                'tags': quote.xpath('div/a/text()').getall(),
            }

        try:
            # Select the NEXT button and store it in next_page
            # Here we include the class of the li tag in the XPath
            # to avoid the difficujlty with the "previous" button
            next_page = response.xpath('/html/body/div/div[2]/div[1]/nav/ul/li[@class="next"]/a').attrib["href"]
        except KeyError:
            # In the last page, there won't be any "href" and a KeyError will be raised
            logging.info('No next page. Terminating crawling process.')
        else:
            # If a next page is found, execute the parse method once again
            yield response.follow(next_page, callback=self.parse)

# Name of the file where the results will be saved
filename = "3_quotesmultiplepages.json"

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
        'src/' + filename: {"format": "json"},
    }
})

# Start the crawling using the spider you defined above
process.crawl(QuotesMultipleSpider)
process.start()