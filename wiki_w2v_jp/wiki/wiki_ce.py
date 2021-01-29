import os
import pickle
from typing import Dict, List, Any
from logging import getLogger

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

logger = getLogger(__name__)


class WikiCE:
    def __init__(self, config: Dict[str, Any]):
        self._val_set: pd.DataFrame = pd.read_csv(os.path.join(config.get('val_path'), "jwsan-1400-val.csv"))
        self._wiki_pkl_path: str = config.get('wiki_pkl_path')
        self._wiki_output_path: str = config.get('wiki_output_path')
        self._n_epochs: int = int(config.get('n_epochs'))
        self._lr: float = float(config.get('learning_rate'))
        self._dim: int = int(config.get('dim'))
        self._noise: int = int(config.get('noise'))
        self._seed: int = int(config.get('seed'))
        with open(os.path.join(config.get('wiki_output_path'), "negative_sample.pkl"), 'rb') as f:
            self._negative_sample: List[str] = pickle.load(f)
        with open(os.path.join(self._wiki_output_path, "wiki_counter.pkl"), 'rb') as f:
            counter = pickle.load(f)
        np.random.seed(self._seed)
        self._tar = {x: np.random.randn(self._dim)*0.01 for x in counter.keys()} # TODO: better init
        self._con = {x: np.random.randn(self._dim)*0.01 for x in counter.keys()}
        del counter
        self._negative_sample_scale = len(self._negative_sample)
    
    def _update_window(self, window: List[str]):
        r = self._tar[window[2]] # target vector
        
        for ix in [0, 1, 3, 4]:
            # Gradient with respect to  $\hat{r}$
            neg_index = np.random.randint(self._negative_sample_scale, size=self._noise)
            w_i = self._con[window[ix]]
            p_i = 1/(1 + np.exp(-w_i.dot(r))) 
            dr = -(1-p_i)*w_i
            for k in neg_index:
                random_string = self._negative_sample[k]
                w_k = self._con[random_string]
                q_k = 1/(1 + np.exp(w_k.dot(r))) 
                dr += (1-q_k) * w_k
                # Gradient with respect to $w_j$
                self._con[random_string] -= (1-q_k) * r * self._lr
            self._con[window[ix]] -= -(1-p_i) * r * self._lr
            self._tar[window[2]] -= dr * self._lr
    
    def _evaluate(self):
        result = []
        for ix in range(len(self._val_set)):
            a_vec = self._tar[self._val_set['word1'][ix]].reshape(1,-1)
            b_vec = self._tar[self._val_set['word2'][ix]].reshape(1,-1)
            result.append(cosine_similarity(a_vec, b_vec)[0][0])
        logger.info(f"the correlation on val is {np.cov(result, self._val_set['similarity'])[0][1]}")
    
    def _fit(self):
        for file in os.listdir(self._wiki_pkl_path):
            with open(os.path.join(self._wiki_pkl_path, file), 'rb') as f:
                batch = pickle.load(f)
                for art_num in range(len(batch)):
                    if art_num % 100 == 0:
                        self._evaluate()
                    article = batch[art_num]
                    for ix in range(len(article)-4):
                        window = article[ix:ix+5]
                        self._update_window(window)
                with open(f"{file}_{art_num}_embedding.pkl", 'wb') as f:
                    pickle.dump([self._tar, self._con], f)
