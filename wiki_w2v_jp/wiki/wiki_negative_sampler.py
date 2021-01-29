import os
from typing import Dict, Any
from logging import getLogger
from collections import Counter

import pandas as pd
import numpy as np
import pickle

logger = getLogger(__name__)


class WikiNegativeSampler:
    def __init__(self, config: Dict[str, Any]):

        self._wiki_output_path: str = config.get('wiki_output_path')
        self._alpha: float = float(config.get('alpha'))
        self._seed: int = int(config.get('seed'))
        self._sample_scale: int = int(config.get('sample_scale'))

    def _sample(self):
        with open(os.path.join(self._wiki_output_path, "wiki_counter.pkl"), 'rb') as f:
            counter = pickle.load(f)

        len_counter = len(counter)
        sum_counter = sum(counter.values())
        logger.info(f'the corpus has {len_counter} unique tokens, summed up with {sum_counter} tokens.')
        sum_counter_alpha = sum_counter ** self._alpha
        prob = {x: counter[x] ** self._alpha / sum_counter_alpha 
            for x 
            in counter.keys()} 
        del counter
        sum_prob = sum(prob.values())
        prob = {x: prob[x] / sum_prob 
            for x 
            in prob.keys()} 

        np.random.seed(self._seed)
        negative_sample_scale = len_counter * self._sample_scale
        negative_sample = list(
            np.random.choice(
                list(prob.keys()), 
                negative_sample_scale, 
                p=list(prob.values())))

        with open(os.path.join(self._wiki_output_path, f"negative_sample.pkl"), 'wb') as f:
            pickle.dump(negative_sample, f)
        logger.info(f'the length of negative_sample is {len(negative_sample)}')
