import collections
import tensorflow as tf
import numpy as np
import io
import json


with io.open('/Users/shingyeong-eun/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/dict.json', 'r') as f:
    dictionary = json.loads(f.read())

with io.open('/Users/shingyeong-eun/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/rdict.json', 'r') as t:
    rdict = json.loads(t.read())

