import tensorflow as tf
import io
import json
import os
from im2txt.ops import image_processing
from models import make_image_embeddings_cnn
import numpy as np


'''
1. 이미지 불러오기
2. 이미지 디코딩
3. 스킵그램 페어
'''

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
                                                       resize_height=346,
                                                       resize_width=346,
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
for image, metadata in image_and_metadata:
    for words in metadata[2]:
        skip_gram_pairs.append((image, words))
        for word in words:
            if word in word2id:
                labels.append(word2id[word])
            else:
                labels.append(len(word2id) + 1)

#labels
print(labels)

labels = tf.placeholder(dtype=tf.int32, shape=[None, voc_size])

#image_embedding
image_embedding = make_image_embeddings_cnn.make_image_embeddings_cnn(decoded_images, voc_size)
print(image_embedding)


#train
nce_weights = tf.Variable(
    tf.random_uniform([image_embedding.shape.as_list()[0], image_embedding.shape.as_list()[1]],
                      -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([image_embedding.shape.as_list()[0]]))

nce_loss = tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                 inputs=image_embedding, labels=labels,
                 num_sampled=14, num_classes=image_embedding.shape.as_list()[0],
                 name='nce_loss')

loss = tf.reduce_mean(nce_loss)

train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
init = tf.global_variables_initializer()

#session
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(image_embedding))
    print(sess.run(loss))
    sess.run(train_op, feed_dict={labels:labels})