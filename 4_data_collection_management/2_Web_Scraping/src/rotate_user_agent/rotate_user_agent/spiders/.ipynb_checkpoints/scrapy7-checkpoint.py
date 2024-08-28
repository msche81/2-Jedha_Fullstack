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
    name = "rotate_user_agent"

    # Url to start your spider from 
    start_urls = [
        'http://quotes.toscrape.com/page/1/',
    ]

    # Callback that gets text, author and tags of the webpage
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
filename = "7_rotate_user_agent.json"

# If file already exists, delete it before crawling (because Scrapy will 
# concatenate the last and new results otherwise)

if filename in os.listdir('results/'):
        os.remove('results/' + filename)