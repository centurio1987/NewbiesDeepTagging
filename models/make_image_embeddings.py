import tensorflow as tf
from im2txt.ops import image_embedding

def make_image_embeddings(images):
    image_embeddings = image_embedding.inception_v3(images,
                                 trainable=False,
                                 is_training=True)

    return image_embeddings