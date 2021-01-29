import os
import gc
import pickle
import json
from collections import Counter
from typing import Dict, List, Any
from logging import getLogger

from fugashi import Tagger
import numpy as np
from unidic import DICDIR
import pandas as pd
from tqdm import tqdm

logger = getLogger(__name__)


class WikiTokenizer:
    def __init__(self, config: Dict[str, Any]):

        self._wiki_text_path: str = config.get('wiki_text_path')
        self._wiki_pkl_path: str = config.get('wiki_pkl_path')
        self._wiki_output_path: str = config.get('wiki_output_path')
        self._check_list: List[str] = config.get('check_list')
        self._tagger: Tagger = Tagger(f'-d "{DICDIR}"')
   
    def _token_filter(self, text: str) -> List[str]:
        parse = [line.split('\t') for line in self._tagger.parse(text).split('\n')[:-1]]
        keep = [line[1].split(',')[0] in self._check_list for line in parse]
        result = [parse[ix][0] if keep[ix] else None for ix in range(len(keep))]
        return list(filter(None, result))

    def _generate_batch(self):
        counter = Counter()
        for directory in os.listdir(self._wiki_text_path):
            path = os.path.join(self._wiki_text_path, directory)
            token_text = []
            for file in tqdm(os.listdir(path)):
                with open(os.path.join(path, file)) as f:
                    for line in f:
                        j_content = json.loads(line)
                        text = self._token_filter(j_content['text'])
                        token_text.append(text)
            counter += Counter([token for article in token_text for token in article])
            with open(os.path.join(self._wiki_pkl_path ,f"{path.split('/')[-1]}.pkl"), 'wb') as f:
                pickle.dump(token_text, f)
        with open(os.path.join(self._wiki_output_path, f"wiki_counter.pkl"), 'wb') as f:
            pickle.dump(counter, f)

