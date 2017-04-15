import tensorflow as tf
import io
import json
import os
from im2txt.ops import image_processing
from models import make_image_embeddings_cnn
import numpy as np
from sklearn.preprocessing import OneHotEncoder


'''
1. 이미지 불러오기
2. 이미지 디코딩
3. 스킵그램 페어
'''

def one_hot(dim, x):
    return np.identity(dim)[x :x+1]

def load_images(image_metadata_list):
    encoded_images = []
    extracted_metadata_by_exist = []
    for image_metadata in image_metadata_list:
        filename = os.path.join("/Users/KYD/Downloads/30_collected_train2014", image_metadata[1])
        if tf.gfile.Exists(filename):
            with tf.gfile.FastGFile(filename, 'rb') as f:
                encoded_images.append(f.read())
                extracted_metadata_by_exist.append(image_metadata)

    return encoded_images, extracted_metadata_by_exist

def decode_images(encoded_images):
    decoded_images = []
    for image in encoded_images:
        decoded_image = image_processing.process_image(image,
                                                       is_training=True,
                                                       thread_id=0,
                                                       height=299,
                                                       width=299,
                                                       resize_height=299,
                                                       resize_width=299,
                                                       image_format='jpeg')
        decoded_images.append(decoded_image)

    return decoded_images

def image_data_mapper(image, metadata):
    return image, metadata

image_metadata_list = []
with io.open('/Users/KYD/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/img_metadata_train.json') as f:
    image_metadata_list = json.loads(f.read())

#voc_size
word2id = dict()
with io.open('/Users/KYD/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/word_to_index_dict.json') as f:
    word2id = json.loads(f.read())
voc_size = len(word2id)

#이미지 불러오기
encoded_images, extracted_metadata = load_images(image_metadata_list)

#이미지 디코딩
decoded_images = decode_images(encoded_images)

image_and_metadata = []
#이미지 메타데이터 페어
for image, metadata in map(image_data_mapper, decoded_images, extracted_metadata):
    image_and_metadata.append((image,metadata))

#skip-gram pair
skip_gram_pairs = []
labels = []
images = []
for image, metadata in image_and_metadata:
    for words in metadata[2]:
        skip_gram_pairs.append((image, words))
        for word in words:
            if word in word2id:
                labels.append([word2id[word]])
                images.append(image)
                skip_gram_pairs.append((image, word2id[word]))
            else:
                labels.append([len(word2id) + 1])
                images.append(image)
                skip_gram_pairs.append((image, len(word2id) + 1))

#labels = tf.one_hot(labels, voc_size)

#one_hot_labels = []
#for i in labels:
#    one_hot_labels.append(one_hot(len(word2id)+1, i))

X = tf.Variable(images)
Y = tf.Variable(labels)
#X = tf.placeholder(dtype=tf.float32, shape=[None, 299, 299, 3])
#Y = tf.placeholder(dtype=tf.int32, shape=[None, 1])

#image_embedding
image_embedding = make_image_embeddings_cnn.make_image_embeddings_cnn(X, image_size=24, voc_size=voc_size)
print(image_embedding)

#train
nce_weights = tf.Variable(
    tf.random_uniform([image_embedding.shape.as_list()[0], image_embedding.shape.as_list()[1]],
                      -1.0, 1.0))

nce_biases = tf.Variable(tf.zeros(image_embedding.shape.as_list()[0]))

nce_loss = tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                 inputs=image_embedding, labels=Y,
                 num_sampled=14, num_classes=voc_size,
                 name='nce_loss')

cost = tf.reduce_mean(nce_loss)

'''
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=image_embedding, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(image_embedding, 1)
'''
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)

init = tf.global_variables_initializer()

#session
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(image_embedding))
    for i in range(3):
        print(sess.run(cost))
        print(sess.run(train_op))