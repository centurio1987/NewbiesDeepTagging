import tensorflow as tf

def make_skip_gram_for_image(images_captions_pairs):
    '''
    images_captions_pairs: [[image, captions], ...]
    image: [width, height, [channel]]
    captions: [[word1, word2, word3], [word1, word2, word3, word4], ...]

    return: [(image, word1), (image, word2), ....]
    '''
    image_word_pairs = []
    for image, captions in images_captions_pairs:
        for word in captions:
            images_captions_pairs.append((image, word))

    return image_word_pairs
