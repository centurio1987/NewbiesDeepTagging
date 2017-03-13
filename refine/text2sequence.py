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
        tagged = nltk.pos_tag(elem)
        refined_tag = []
        for item in tagged:
            if item[1] == 'FW' or item[1] == 'JJ' or item[1] == 'NN'\
                or item[1] == 'NNS' or item[1] == 'NNP' or item[1] == 'NNPS'\
                or item[1] == 'VB' or item[1] == 'VBD' or item[1] == 'VBG'\
                or item[1] == 'VBN' or item[1] == 'VBP' or item[1] == 'VBZ':
                refined_tag.append(item)

        postagged_caption_list.append(refined_tag)

    with io.open('/Users/KYD/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/nltk_test.json', 'w') as ff:
        json_result = json.dumps(postagged_caption_list)
        ff.write(json_result)
