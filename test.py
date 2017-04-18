import tensorflow as tf
import io
import json
import os
from im2txt.im2txt.ops import image_processing
from models import make_image_embeddings_cnn
import numpy as np

#flags
test_input_dir = "/home/ubuntu/data/images/10_collected_train2014"
metadata_file = '/home/ubuntu/data/metadata/img_metadata_train.json'
word2id_file = '/home/ubuntu/data/metadata/word_to_index_dict.json'

def one_hot(dim, x):
    return np.identity(dim)[x :x+1]

def load_images(image_metadata_list):
    encoded_images = []
    extracted_metadata_by_exist = []
    for image_metadata in image_metadata_list:
        filename = os.path.join(test_input_dir, image_metadata[1])
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
with io.open(metadata_file) as f:
    image_metadata_list = json.loads(f.read())

#voc_size
word2id = dict()
with io.open(word2id_file) as f:
    word2id = json.loads(f.read())
word2id['NONE'] = len(word2id)
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
        for word in words:
            if word in word2id:
                labels.append([word2id[word]])
                images.append(image)
                skip_gram_pairs.append([image, word2id[word]])
            else:
                labels.append([word2id['NONE']])
                images.append(image)
                skip_gram_pairs.append([image, word2id['NONE']])

#labels = tf.one_hot(labels, voc_size)

#one_hot_labels = []
#for i in labels:
#    one_hot_labels.append(one_hot(len(word2id)+1, i))

#X = tf.placeholder(dtype=tf.float32, shape=[None, 299, 299, 3])
#Y = tf.placeholder(dtype=tf.int32, shape=[None, 1, voc_size])

w1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01), name='w1')
w2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01), name='w2')
w3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01), name='w3')
w4 = tf.Variable(tf.random_normal([128 * 38 * 38, 625], stddev=0.01), name='w4')
p_keep_conv = 1
p_keep_hidden = 1

image_embedding = make_image_embeddings_cnn.model(images, w1, w2, w3, w4, p_keep_conv, p_keep_hidden)

w_o = tf.Variable(tf.random_normal(shape=[voc_size, image_embedding.shape.as_list()[1]]), name='output_weight')
bias = tf.Variable(tf.zeros([voc_size]), name='bias')

logits = tf.matmul(image_embedding, tf.transpose(w_o))
logits = tf.nn.bias_add(logits, bias)
labels_one_hot = tf.one_hot(labels, voc_size)
soft_cost = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels_one_hot,
      logits=logits)
cost = tf.reduce_sum(soft_cost)

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'saved_model/my-model.ckpt')

    trained_image_embedding = image_embedding.eval()
    
