import tensorflow as tf
import io
import json
import os
from im2txt.im2txt.ops import image_processing
from models import make_image_embeddings_cnn
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''
#1. index2filename 리스트 만들기
#2. 이미지 셋에 대하여 이미지 임베딩 만들기
#3. 지정한 인덱스에 대해 전 이미지와 코사인 거리 구하기
#4. 가장 가까운 이미지의 로짓을 구하여 가장 크기가 큰
'''

#flags
test_input_dir = "/home/ubuntu/data/images/test_for_1000_train"
metadata_file = '/home/ubuntu/data/metadata/img_metadata_train.json'
word2id_file = '/home/ubuntu/data/metadata/word_to_index_dict.json'
id2word_file = '/home/ubuntu/data/metadata/index_to_word_list.json'

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

def make_idx_to_imagefile(image_metadata):
    idx2imgfile_list = []
    for metadata in image_metadata:
        filename = os.path.join(test_input_dir, metadata[1])
        if tf.gfile.Exists(filename):
            idx2imgfile_list.append(filename)

    return idx2imgfile_list

#파일로부터 데이터 불러오기
image_metadata_list = []
with io.open(metadata_file) as f:
    image_metadata_list = json.loads(f.read())

#voc_size
word2id = dict()
with io.open(word2id_file) as f:
    word2id = json.loads(f.read())
word2id['NONE'] = len(word2id)
voc_size = len(word2id)

id2word = []
with io.open(id2word_file) as f:
    id2word = json.loads(f.read())
id2word.append(None)

#이미지 불러오기
encoded_images, extracted_metadata = load_images(image_metadata_list)

#idx to image file name list
idx2imgfile_list = make_idx_to_imagefile(extracted_metadata)

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

unique_images = list(set(images))
#labels = tf.one_hot(labels, voc_size)

#one_hot_labels = []
#for i in labels:
#    one_hot_labels.append(one_hot(len(word2id)+1, i))

X = tf.placeholder(dtype=tf.float32, shape=[None, 299, 299, 3])
Y = tf.placeholder(dtype=tf.int32, shape=[None, 1])

w1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01), name='w1')
w2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01), name='w2')
w3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01), name='w3')
w4 = tf.Variable(tf.random_normal([128 * 38 * 38, 625], stddev=0.01), name='w4')
p_keep_conv = 1
p_keep_hidden = 1

image_embedding = make_image_embeddings_cnn.model(X, w1, w2, w3, w4, p_keep_conv, p_keep_hidden)

w_o = tf.Variable(tf.random_normal(shape=[voc_size, image_embedding.shape.as_list()[1]]), name='output_weight')
bias = tf.Variable(tf.zeros([voc_size]), name='bias')

logits = tf.matmul(image_embedding, tf.transpose(w_o))
logits = tf.nn.bias_add(logits, bias)
softmaxed_logits = tf.nn.softmax(logits=logits)
'''
labels_one_hot = tf.one_hot(labels, voc_size)
soft_cost = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels_one_hot,
      logits=logits)
cost = tf.reduce_sum(soft_cost)
'''
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'saved_model/my-model.ckpt')
    batch_size = 10
    trained_image_embedding = []
    trained_logits = []
    trained_softmaxed_logits = []
    for start in range(0, len(unique_images), batch_size):
        end = start + batch_size
        if end > len(images):
            end = len(images)
        batch_images = images[start:end]
        x_data_list = []
        for image in batch_images:
            x_data_list.append(image.eval())

        trained_image_embedding[start:end] = image_embedding.eval(feed_dict={X:x_data_list})
        trained_logits[start:end] = logits.eval(feed_dict={X:x_data_list})
        trained_softmaxed_logits[start:end] = softmaxed_logits.eval(feed_dict={X:x_data_list})

#타겟 이미지에 대해 코사인 거리 구하기
test_target_image_idx = 5
cosine_result_list = []

# trained_image_embedding's shape is (130, 625)

for i in range(len(trained_image_embedding)):
    cosine_result_list.append(abs(cosine(trained_image_embedding[test_target_image_idx], trained_image_embedding[i])))

matched_image_idx = np.argmax(cosine_result_list)
'''
plt_img = mpimg.imread(idx2imgfile_list[matched_image_idx])
plt.imshow(plt_img)
plt.show()
'''
print('test_target_image_file: ', idx2imgfile_list[test_target_image_idx])
print('matched_image_file: ', idx2imgfile_list[matched_image_idx])
'''
print('softmax_logits: ', trained_softmaxed_logits)
print('softmax_logits\'s shape: ', trained_softmaxed_logits[0].shape)
print('voc_size: ', voc_size)
'''

picked_idxs = np.argsort(trained_softmaxed_logits[matched_image_idx])[-10:][::-1]
print('nearest picture\'s top 10 words: ', [id2word[x] for x in picked_idxs])

picked_idxs_for_test = np.argsort(trained_softmaxed_logits[test_target_image_idx])[-10:][::-1]
print('test picture\'s top 10 words: ', [id2word[x] for x in picked_idxs_for_test])
