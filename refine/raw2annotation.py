import json
import io

with io.open('/Users/shingyeong-eun/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/captions_val2014.json', 'r') as f:
    dict1 = json.loads(f.read())
    ex = dict1["annotations"]
    with io.open('/Users/shingyeong-eun/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/img_metadata_for_val.json',
                 'w') as t:
        dump = json.dumps(ex)
        t.write(dump)
