import tensorflow as tf
from models import input_data_builder

embedding_size = 512
num_sampled = 15
batch_size = 256

input_builder = input_data_builder.InputDataBuilder()
input_builder.build_input()

nce_weights = tf.Variable(
    tf.random_uniform([input_builder.image_embeddings.shape.as_list()[0], input_builder.image_embeddings.shape.as_list()[1]],
                      -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([input_builder.image_embeddings.shape.as_list()[0]]))

nce_loss = tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                 inputs=input_builder.image_embeddings, labels=input_builder.batch_contexts,
                 num_sampled=num_sampled, num_classes=input_builder.image_embeddings.shape.as_list()[0],
                 name='nce_loss')

loss = tf.reduce_mean(nce_loss)

train_op = tf.train.AdamOptimizer(0.01).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print('image_embeddings',sess.run(input_builder.image_embeddings))
    print('nce_weights',sess.run(nce_weights))
    print('nce_loss', sess.run(nce_loss))
    print('loss', sess.run(loss))
    print(sess.run(train_op))