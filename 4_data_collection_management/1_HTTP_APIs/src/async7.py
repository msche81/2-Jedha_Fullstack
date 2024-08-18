import asyncio
import time
import logging

logger = logging.getLogger(__name__)

async def API_call(url: int):
    await asyncio.sleep(2) # let's say an API call takes 2 seconds to finish
    return "response"

list_of_urls = [i for i in range(100)]

async def main():
    logger.info("Starting query...")
    start = time.time()
    tasks = [asyncio.create_task(API_call(url)) for url in list_of_urls]
    result =  await asyncio.gather(*tasks)
    end = time.time()
    elapsed = str(end - start)
    logger.info("The process took {} seconds".format(elapsed))
    return result