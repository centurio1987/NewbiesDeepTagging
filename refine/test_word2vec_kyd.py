import json
import io
from gensim import corpora

# 파일로부터 text, dictionary 호출
# skip-gram 모델 만들기
# batch-set 만들기
# 임베딩테이블 만들기
# 학습하기

dic = dict()
rdic = []

with io.open('/Users/KYD/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/dict.json', 'r') as f:
    dic = json.loads(f.read())

with io.open('/Users/KYD/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/rdict.json', 'r') as f:
    rdic = json.loads(f.read())

