import tensorflow as tf

image = tf.read_file('/Users/shingyeong-eun/Dropbox/논문/MSCOCO/captions_train-val2014/sample_img/1.jpg')
decoded_img = tf.image.decode_jpeg(image)
resized_img = tf.image.resize_images(decoded_img, 25, 25)

with tf.Session() as sess:
    result = sess.run(resized_img)
    print(result.shape)