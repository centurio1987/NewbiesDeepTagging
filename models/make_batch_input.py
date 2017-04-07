import tensorflow as tf

def make_batch_input(skip_gram_pairs):
    batch_images, batch_contexts = tf.train.batch_join(skip_gram_pairs,
                        batch_size=256,
                        dynamic_pad=True,
                        name='input_batch')

    return batch_images, batch_contexts