import json
import io
from gensim import corpora

# skip-gram 모델 만들기
# batch-set 만들기
# 임베딩테이블 만들기
# 학습하기

dic = dict()
rdic = []

with io.open('/Users/KYD/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/word_to_index_dict.json', 'r') as f:
    dic = json.loads(f.read())

with io.open('/Users/KYD/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/index_to_word_list.json', 'r') as f:
    rdic = json.loads(f.read())

