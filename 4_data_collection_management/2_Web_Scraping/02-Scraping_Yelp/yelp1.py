import os
import logging

import scrapy
from scrapy.crawler import CrawlerProcess

class YelpSpider(scrapy.Spider):
    # Name of your spider
    name = "yelp"

    # Starting URL
    start_urls = ['https://www.yelp.fr/']

    # Parse function for form request
    def parse(self, response):
        # FormRequest used to make a search in Paris
        return scrapy.FormRequest.from_response(
            response,
            formdata={'find_desc': 'restaurant japonais', 'dropperText_Mast': 'paris'},
            callback=self.after_search
        )

    # Callback used after login
    #def after_search(self, response):
        
        #names = response.xpath('//*[@id="main-content"]/ul/li[3]/div[1]/div/div[2]/div[1]/div[1]/div[1]/div/div/h3/a/text()')
        #urls = response.xpath('//*[@id="main-content"]/ul/li[3]/div[1]/div/div[2]/div[1]/div[1]/div[1]/div/div/h3/a')
        
        #for name, url in zip(names,urls):
            #yield {
                #'name': name.get(),
                #'url': url.attrib["href"]
            #}

   # Callback used after login
    def after_search(self, response):
    # Note: supprime l'index [3] pour sélectionner tous les éléments

        names = response.xpath('//*[@id="main-content"]/ul/li/div[1]/div/div[2]/div[1]/div[1]/div[1]/div/div/h3/a/text()').getall()
        urls = response.xpath('//*[@id="main-content"]/ul/li/div[1]/div/div[2]/div[1]/div[1]/div[1]/div/div/h3/a/@href').getall()
    
        for name, url in zip(names, urls):
            yield {
                'name': name,
                'url': url
            }

# Name of the file where the results will be saved
filename = "restaurant_japonais-paris.json"

# If file already exists, delete it before crawling (because Scrapy will concatenate the last and new results otherwise)
if filename in os.listdir('02-Scraping_Yelp/'):
        os.remove('02-Scraping_Yelp/' + filename)

# Declare a new CrawlerProcess with some settings
process = CrawlerProcess(settings = {
    'USER_AGENT': 'Chrome/97.0',
    'LOG_LEVEL': logging.INFO,
    "FEEDS": {
        '02-Scraping_Yelp/' + filename: {"format": "json"},
    }
})

# Start the crawling using the spider you defined above
process.crawl(YelpSpider)
process.start()