import tensorflow as tf

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def make_image_embeddings_cnn(images, image_size, voc_size):
    w = init_weights([3, 3, 3, 32])  # 3x3x1 conv, 32 outputs
    w2 = init_weights([3, 3, 32, 64])  # 3x3x32 conv, 64 outputs
    w3 = init_weights([3, 3, 64, 128])  # 3x3x32 conv, 128 outputs
    w4 = init_weights([128 * 38 * 38, 625])  # FC 128 * 4 * 4 inputs, 625 outputs
    w_o = init_weights([625, voc_size])  # FC 625 inputs, 10 outputs (labels)

    p_keep_conv = 0.8
    p_keep_hidden = 0.5
    image_embeddings = model(images, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)
    return image_embeddings

def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx
