import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

image = tf.read_file('/Users/KYD/Dropbox/논문/MSCOCO/captions_train-val2014/sample_img/1.jpg')
decoded_image = tf.image.decode_jpeg(image)
resized_image = tf.image.resize_images(decoded_image, 30, 30)

with tf.Session() as sess:
    result = sess.run(resized_image)
    print(result.shape)