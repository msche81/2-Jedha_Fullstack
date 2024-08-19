"""
youtube_api.py

Some utility variables and function to run the data collection.
"""
import os
from typing import Sequence
from urllib.parse import urlencode

BASE_URL = 'https://www.googleapis.com/youtube/v3/videos'
API_KEY = os.environ['YOUTUBE_API_KEY']
PARTS = [
    'contentDetails', 'snippet',
    'statistics', 'status', 'topicDetails'
]


def build_query_parameters(video_ids: Sequence) -> str:
    parameters = {
        'part': ','.join(PARTS),
        'id': ','.join(video_ids),
        'key': API_KEY
    }
    return urlencode(parameters)
