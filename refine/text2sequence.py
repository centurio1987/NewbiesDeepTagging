import nltk
import io
import json

with io.open('/Users/KYD/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/val_test.json', 'r') as f:
    annotation = json.loads(f.read())
    caption_list = []
    for elem in annotation:
        caption_list.append(elem['caption'])

    tokenized_caption_list = []
    for elem in caption_list:
        tokenized_caption_list.append(nltk.word_tokenize(elem))

    postagged_caption_list = []
    for elem in tokenized_caption_list:
        postagged_caption_list.append(nltk.pos_tag(elem))

    with io.open('/Users/KYD/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/nltk_test.json', 'w') as ff:
        json_result = json.dumps(postagged_caption_list)
        ff.write(json_result)
