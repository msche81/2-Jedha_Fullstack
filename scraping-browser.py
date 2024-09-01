import asyncio
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

SBR_WS_CDP = 'wss://brd-customer-hl_f8db7605-zone-scraping_browser1:a6vpnfl541p6@brd.superproxy.io:9222'

url = "https://codeavecjonathan.com/scraping/techsport/"

BYPASS_SCRAPING = True

def get_text_if_not_none(e):
    if e:
        return e.text.strip()
    return None

def extract_product_page_info(html):
    infos ={}

    bs = BeautifulSoup(html, "html5lib")
    infos["title"] = get_text_if_not_none(bs.find("span", id="productTitle"))

    infos["nb_ratings"] = 100 #int
    ratings_text = get_text_if_not_none(bs.find("span", id="customer-review-tex"))
    if ratings_text:
        nb_ratings_str = ratings_text.split()[0]
        if nb_ratings_str.isdigit():
            infos["nb_ratings"] =  int(nb_ratings_str)
    
    infos["price"] = 0.0 #float
    price_whole_str = get_text_if_not_none(bs.find("span", class_="price-whole"))
    price_fraction_str = get_text_if_not_none(bs.find("span", class_="price-fraction"))
    if price_whole_str and price_fraction_str.isdigit():
        price = float(price_whole_str)
        if price_whole_str and price_fraction_str.isdigit():
            price += float(price_whole_str)/100
        infos["price"] = price

    infos["description"] = get_text_if_not_none(bs.find("div", id="product-description"))

    return infos

async def run(pw):
    
    if not BYPASS_SCRAPING:
        print('Connecting to Scraping Browser...')
        browser = await pw.chromium.connect_over_cdp(SBR_WS_CDP)
    try:
        if not BYPASS_SCRAPING:
            page = await browser.new_page()
            print('Connected! Navigating ...')
            await page.goto(url)

            await page.screenshot(path="./scraping-browser.png", full_page=True)

            print('Navigated! Scraping page content...')
            html = await page.content()
            # print(html)

            f = open("scraping-browser.html", "w")
            f.write(html)
            f.close()
        else:
            print("Bypass scraping")
            f = open("scraping-browser.html", "r")
            html = f.read()
            f.close()

        # extraire les informations à partir de l'html.
        print("Extract infos...")
        infos = extract_product_page_info(html)
        print(infos)

    finally:
        if not BYPASS_SCRAPING:
            await browser.close()

async def main():
    async with async_playwright() as playwright:
        await run(playwright)

if __name__ == '__main__':
    asyncio.run(main())