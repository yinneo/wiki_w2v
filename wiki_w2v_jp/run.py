from logging import basicConfig, getLogger, DEBUG
from typing import Any

import yaml

from wiki import WikiTokenizer, WikiNegativeSampler, WikiCE

logger = getLogger(__name__)


def main():
    logger.info('Loading config.yml')
    with open('config.yml') as f:
        config: Any = yaml.load(f, Loader=yaml.SafeLoader)
    
    logger.info('Starting WikiTokenizer.')
    tokenizer: WikiTokenizer = WikiTokenizer(config)._generate_batch()
    logger.info('Starting to prepare the negative samples.')
    sampler: WikiNegativeSampler = WikiNegativeSampler(config)._sample()
    logger.info('Starting CE training process.')
    wikice: WikiCE = WikiCE(config)._fit()


if __name__ == "__main__":
    basicConfig(filename='val.log', level=DEBUG)
    main()
