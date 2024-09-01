import sys
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import urllib.request

url = "https://codeavecjonathan.com/scraping/techsport/"

opener = urllib.request.build_opener(
    urllib.request.ProxyHandler(
        {'http': 'http://brd-customer-hl_f8db7605-zone-web_unlocker1:2jx0w3jd50ff@brd.superproxy.io:22225',
        'https': 'http://brd-customer-hl_f8db7605-zone-web_unlocker1:2jx0w3jd50ff@brd.superproxy.io:22225'}))

html = opener.open(url).read().decode("utf-8")

f = open("web-unlocker.html", "w")
f.write(html)
f.close()