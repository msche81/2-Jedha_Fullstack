"""
data_collection.py

Asychronously collect extra data from the Youtube API.

@author:
Edouard Theron <edouard@nibble.ai>
"""
import asyncio
from concurrent import futures
import logging
import json
import time
import random
from typing import Optional, Sequence
from uuid import uuid4

import requests
from requests.exceptions import RequestException
from tornado.httpclient import AsyncHTTPClient

from config import config
import youtube_api

logger = logging.getLogger(__name__)



def get_batch(video_ids: Sequence, batch_size: int = 50) -> Sequence:
    """Yield the next batch of 50 videos given a sequence of videos IDs."""
    for i in range(0, len(video_ids), batch_size):
        yield video_ids[i:i + batch_size]


async def fetch_batch(ids: Sequence, http_client: AsyncHTTPClient, 
                      dry_run: bool) -> Optional[dict]:
    """Make a single API call and return the result.
    
    We allow for a `dry_run` mode to avoid spending API quotas when testing.
    If the actual API call succeeded, return the JSON data sent as a response.
    Else, return None.
    """
    assert len(ids) <= 50, 'Maximum batch size is 50 videos at a time.'
    
    if dry_run:
        # Simulate a blocking API call to preserve our API quotas
        await asyncio.sleep(1 + random.random())
        return None

    query = youtube_api.build_query_parameters(ids)
    url = f'{youtube_api.BASE_URL}?{query}'
    resp = await http_client.fetch(url)
    if resp.code != 200:
        try:
            msg = json.loads(resp.body.decode())['error']['message']
        except (AttributeError, KeyError, TypeError):
            msg = 'Unknown cause.'
        logger.error(f'Call failed for batch {ids[0]}...{ids[-1]}: '
                     f'{resp.code} "{msg}"')
        return None
    logger.debug('API call succeeded.')
    data = json.loads(resp.body.decode())
    return data


async def fetch_all(ids: Sequence, dry_run: bool = True) -> list:
    """Fetch data asynchronously and save aggregated results in a file."""
    start = time.time()
    client = AsyncHTTPClient()
    logger.info(f'Requesting data for {len(ids)} videos. Please wait...')
    tasks = [
        asyncio.create_task(fetch_batch(batch_ids, client, dry_run))
        for batch_ids in get_batch(ids)
    ]
    all_data = await asyncio.gather(*tasks)
    
    duration = time.time() - start
    log_msg = f'Done! Fetched data from {len(ids)} videos in {duration:.2f} sec'
    if dry_run:
        logger.debug(f'[DRY RUN] {log_msg}')
    else:
        logger.info(log_msg)

    return all_data

async def fetch_all_and_store(ids: Sequence, dry_run: bool = True) -> list:
    """Fetch data asynchronously and save aggregated results in a file."""
    start = time.time()
    client = AsyncHTTPClient(connect_timeout =0,
                             request_timeout =0)
    logger.info(f'Requesting data for {len(ids)} videos. Please wait...')
    tasks = [
        asyncio.create_task(fetch_batch(batch_ids, client, dry_run))
        for batch_ids in get_batch(ids)
    ]
    all_data = await asyncio.gather(*tasks)
    
    duration = time.time() - start
    log_msg = f'Done! Fetched data from {len(ids)} videos in {duration:.2f} sec'
    if dry_run:
        logger.debug(f'[DRY RUN] {log_msg}')
    else:
        logger.info(log_msg)
    


    return all_data


##############################################################################
#                             ALTERNATIVE METHOD                             #
##############################################################################
# Prior to asyncio, we would have performed our asynchronous tasks with the 
# `concurrent.futures` package, creating a pool of threads to make our 
# thousands of API calls. The performance is about the same as with `asyncio`
# but this method uses a little more memory. Whenever it's possible, always
# prefer the `asyncio` method.

# Here is a working sample.

def alt_fetch_batch(ids: Sequence, dry_run: bool) -> Optional[dict]:
    """Given a list of video IDs, return information from Youtube API."""
    assert len(ids) <= 50, 'No more than 50 videos at a time.'

    logger.debug(f'Fetching info for {ids[0]}...')
    if dry_run:
        # Simulate a blocking network I/O to avoid using our API quotas
        time.sleep(1.5 + random.random())
        return

    query = youtube_api.build_query_parameters(ids)
    url = f'{youtube_api.BASE_URL}?{query}'
    try:
        resp = requests.get(url)
    except RequestException as err:
        logger.error(f'Error when fetching info for {ids[0]}...{ids[-1]}: '
                     f'{err}')
        return

    if resp.status_code != 200:
        try:
            msg = resp.json()['error']['message']
        except (AttributeError, KeyError):
            msg = 'Unknown cause.'
        logger.error(f'Call failed for batch {ids[0]}...{ids[-1]}: '
                     f'{resp.status_code} "{msg}"')
        return

    logger.debug(f'Info about {ids[0]} fetched.')
    return resp.json()
        

def alt_fetch_all(ids: Sequence, dry_run: bool = True) -> list:
    start = time.time()
    # Depending on your OS and the state of your machine at run time,
    # you may be able to spin up more than 1000 workers.
    # If your OS reaches a limit, an error will be raised.
    with futures.ThreadPoolExecutor(max_workers=1000) as tpe:
        tasks = [
            tpe.submit(alt_fetch_batch, batch_ids, dry_run)
            for batch_ids in get_batch(ids)
        ]

    all_data = [f.result() for f in futures.as_completed(tasks)]

    duration = time.time() - start
    log_msg = f'Fetched data from {len(ids)} videos in {duration:.2f} sec'
    if dry_run:
        logger.info(f'[DRY RUN] {log_msg}')
    else:
        logger.info(log_msg)

    return all_data
##############################################################################
