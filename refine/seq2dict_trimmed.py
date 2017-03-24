import io
import json
from collections import defaultdict
from gensim import corpora

with io.open('/Users/shingyeong-eun/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/tokenized_caption.json', 'r') as t:
    texts = json.loads(t.read())
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1


    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]

    rdict = []
    for k in frequency.keys():
        if frequency[k] > 1:
            rdict.append(k)

    dict = dict()
    for i, w in enumerate(rdict):
        dict[w] = i

    '''
    with io.open('/Users/shingyeong-eun/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/trimmed_tokenized_caption.json',
     'w') as t:
        json_result = json.dumps(texts)
        t.write(json_result)


    dictionary = corpora.Dictionary(texts)
    dict = dictionary.token2id

    with io.open('/Users/shingyeong-eun/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/word_to_index_dict.json',
     'w') as tt:
        json_dump = json.dumps(dict)
        tt.write(json_dump)

    with io.open('/Users/shingyeong-eun/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/index_to_word_list.json',
     'w') as ff:
        rdict_dump = json.dumps(rdict)
        ff.write(rdict_dump)
    '''

    #print(dict['stranded'])
    #print(rdict[1])
