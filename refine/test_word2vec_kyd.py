import json
import io
from gensim import corpora

# skip-gram 모델 만들기
# batch-set 만들기
# 임베딩테이블 만들기
# 학습하기

dic = dict()
rdic = []
window_size = 2

with io.open('/Users/KYD/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/word_to_index_dict.json', 'r') as f:
    dic = json.loads(f.read())

with io.open('/Users/KYD/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/index_to_word_list.json', 'r') as f:
    rdic = json.loads(f.read())

index_data = []
with io.open('/Users/KYD/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/trimmed_tokenized_caption.json', 'r') as f:
    trimmed_tokenized_caption = json.loads(f.read())

    for caption in trimmed_tokenized_caption:
        templist = []
        for token in caption:
            templist.append(dic[token])

        index_data.append(templist)

print(index_data[:10])
#skip-gram pair 만들기

skip_gram_pairs = []

for sequence in index_data:
    for target_index in range(len(sequence)):
        for window_index in range(0 - window_size, window_size+1):
            if 0 <= target_index + window_index < len(sequence) and target_index is not window_index + target_index:
                skip_gram_pairs.append([sequence[target_index], sequence[target_index + window_index]])

print(skip_gram_pairs[:10])