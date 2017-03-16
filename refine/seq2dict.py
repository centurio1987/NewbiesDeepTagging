import io
import json
from collections import defaultdict
from gensim import corpora

with io.open('/Users/KYD/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/nltk_test.json', 'r') as t:
    texts = json.loads(t.read())
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]


    dictionary = corpora.Dictionary(texts)
    dictionary.save('/Users/KYD/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/dict_test.dict')

    print(len(dictionary.token2id))