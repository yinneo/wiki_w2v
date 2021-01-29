# Vector Semantics and Embeddings


## Dataset

- Japanese Wikipedia Corpus
- Data Extractor: https://github.com/attardi/wikiextractor
    - Converted into NDJSON format files. (each line corresponds to 1 json object)
- JSON format:

```
{"id": "", "revid": "", "url":"", "title": "", "text": "..."}
```


### Test Data

- [日本語単語類似度・関連度データセット JWSAN](http://www.utm.inf.uec.ac.jp/JWSAN/jwsan-1400.csv)
- This data contain pairs of words and each similarity which is rated by human from 7 levels (0 to 6).

### Evaluation

- Similarity: cosine similarity
- Metric: correlation between similarity in test data and similarity calculated from word embeddings.

