import io
import json

'''
with io.open('/Users/shingyeong-eun/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/captions_train2014.json', 'r') as f:
    dict1 = json.loads(f.read())
    ex = dict1["images"]
    with io.open('/Users/shingyeong-eun/Dropbox/img_train.json', 'w') as t:
        dump = json.dumps(ex)
        t.write(dump)

'''
with io.open('/Users/shingyeong-eun/Dropbox/img_train.json', 'r') as f:
    img = json.loads(f.read())
    with io.open('/Users/shingyeong-eun/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/'
                 'img_metadata_with_tokenized_caption_train.json', 'r') as ff:
        caption = json.loads(ff.read())
    metadata_list = []
    for elem in img:
        semi_list = []
        semi_list.append(elem['id'])
        semi_list.append(elem['file_name'])
        semi_list.append([])
        metadata_list.append(semi_list)

    for i, big in enumerate(metadata_list):
        for dict in caption:
            if dict["image_id"] == big[0]:
                big[2].append(dict["caption"])
        print('ok', (i / len(metadata_list)) * 100)


    with io.open('/Users/shingyeong-eun/Dropbox/img_metadata_train.json', 'w') as t:
        dump = json.dumps(metadata_list)
        t.write(dump)
