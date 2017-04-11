import tensorflow as tf
from im2txt.ops import image_embedding

def make_image_embeddings(images):
    inception_output = image_embedding.inception_v3(images,
                                 trainable=False,
                                 is_training=True)

    initializer = tf.random_uniform_initializer(
        minval=-0.08,
        maxval=0.08)

    # Map inception output into embedding space.
    with tf.variable_scope("image_embedding") as scope:
        image_embeddings = tf.contrib.layers.fully_connected(
            inputs=inception_output,
            num_outputs=512,
            activation_fn=None,
            weights_initializer=initializer,
            biases_initializer=None,
            scope=scope)

    return inception_output, image_embeddings