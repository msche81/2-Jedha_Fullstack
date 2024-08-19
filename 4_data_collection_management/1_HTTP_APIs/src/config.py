"""
Config

@authors:
Florent Piétot <florent@nibble.ai>
Edouard Theron <edouard@nibble.ai>
"""
import logging.config
from pathlib import Path


class Config(object):

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parents[1]
        self.data_dir = self.base_dir / 'input'
        self.log_dir = self.base_dir / 'logs'
        if not self.log_dir.exists():
            self.log_dir.mkdir()


    @property
    def logging(self):
        return {
            'version': 1,
            'disable_existing_loggers': True,
            'formatters': {
                'simple': {
                    'format': '[{asctime}]\t{levelname}\t{message}\t'
                              '({name})',
                    'style': '{',
                    'datefmt': '%Y-%m-%d %H:%M:%S %Z'
                },
            },
            'handlers': {
                'file': {
                    'level': 'INFO',
                    'class': 'logging.FileHandler',
                    'filename': f'{self.log_dir}/app.log',
                    'formatter': 'simple'
                },
                'stream': {
                    'level': 'DEBUG',
                    'class': 'logging.StreamHandler',
                    'formatter': 'simple'
                },
            },
            'loggers': {
                'src': {
                    'handlers': ['file', 'stream'],
                    'level': 'DEBUG',
                    'propagate': False
                },
                '': {
                    'handlers': ['file', 'stream'],
                    'level': 'DEBUG',
                    'propagate': False
                },
            }
        }


config = Config()
logging.config.dictConfig(config.logging)