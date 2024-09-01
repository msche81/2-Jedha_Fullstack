import asyncio from playwright.async_api
import async_playwright
SBR_WS_CDP = 'YOUR_CREDENTIALS'
async def run(pw):
    print('Connecting to Scraping Browser...')
    browser = await pw.chromium.connect_over_cdp(SBR_WS_CDP)
    try:
        page = await browser.new_page()
        print('Connected! Navigating to example.com')
        await page.goto('https://example.com')
        html = await page.content()
        print(html)
    finally:
        await browser.close()

async def main():
    async with async_playwright() as playwright:
        await run(playwright)

if __name__ == '__main__':
    asyncio.run(main())