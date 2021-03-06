import io
import json
import random
import threading
import numpy as np
import sys
import os
import tensorflow as tf
from datetime import datetime

from collections import Counter
from collections import namedtuple
#recommened train_shards: 256
tf.flags.DEFINE_integer("train_shards", 1,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 4,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 8,
                        "Number of shards in testing TFRecord files.")
tf.flags.DEFINE_integer("num_threads", 4,
                        "Number of threads to preprocess the images.")
tf.flags.DEFINE_string("output_dir", "/Users/KYD/Downloads/tfrecord", "Output data directory.")
tf.flags.DEFINE_string("train_image_dir", "/Users/KYD/Downloads/30_collected_train2014",
                       "Training image directory.")

FLAGS = tf.flags.FLAGS

ImageMetadata = namedtuple("ImageMetadata",["image_id", "filename", "captions"])

class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, vocab, unk_id):
        """Initializes the vocabulary.

        Args:
          vocab: A dictionary of word to word_id.
          unk_id: Id of the special 'unknown' word.
        """
        self._vocab = vocab
        self._unk_id = unk_id

    def word_to_id(self, word):
        """Returns the integer id of a word string."""
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._unk_id

class ImageDecoder(object):
    """Helper class for decoding images in TensorFlow."""

    def __init__(self):
    # Create a single TensorFlow Session for all image decoding calls.
        self._sess = tf.Session()

        # TensorFlow ops for JPEG decoding.
        self._encoded_jpeg = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

    def decode_jpeg(self, encoded_jpeg):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._encoded_jpeg: encoded_jpeg})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    if type(value) == str:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def to_sequence_example(image, decoder, vocab):
    """Builds a SequenceExample proto for an image-caption pair.

    Args:
    image: An ImageMetadata object.
    decoder: An ImageDecoder object.
    vocab: A Vocabulary object.

    Returns:
    A SequenceExample proto.
    """
    filename = output_file = os.path.join(FLAGS.train_image_dir, image.filename)
    if tf.gfile.Exists(filename):
        with tf.gfile.FastGFile(filename, "rb") as f:
            encoded_image = f.read()
    else:
        return

    try:
        decoder.decode_jpeg(encoded_image)
    except (tf.errors.InvalidArgumentError, AssertionError):
        print("Skipping file with invalid JPEG data: %s" % image.filename)
        return

    context = tf.train.Features(feature={
      "image/image_id": _int64_feature(image.image_id),
      "image/data": _bytes_feature(encoded_image),
    })

    assert len(image.captions) == 1
    caption = image.captions[0]
    caption_ids = [vocab.word_to_id(word) for word in caption]
    feature_lists = tf.train.FeatureLists(feature_list={
      "image/caption": _bytes_feature_list(caption),
      "image/caption_ids": _int64_feature_list(caption_ids)
    })
    sequence_example = tf.train.SequenceExample(
      context=context, feature_lists=feature_lists)

    return sequence_example

def process_image_files(thread_index, ranges, name, images, decoder, vocab,
                         num_shards):
    """Processes and saves a subset of images as TFRecord files in one thread.

    Args:
    thread_index: Integer thread identifier within [0, len(ranges)].
    ranges: A list of pairs of integers specifying the ranges of the dataset to
      process in parallel.
    name: Unique identifier specifying the dataset.
    images: List of ImageMetadata.
    decoder: An ImageDecoder object.
    vocab: A Vocabulary object.
    num_shards: Integer number of shards for the output files.
    """
    # Each thread produces N shards where N = num_shards / num_threads. For
    # instance, if num_shards = 128, and num_threads = 2, then the first thread
    # would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
    num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in images_in_shard:
          image = images[i]

          sequence_example = to_sequence_example(image, decoder, vocab)
          if sequence_example is not None:
            writer.write(sequence_example.SerializeToString())
            shard_counter += 1
            counter += 1

          if not counter % 1000:
            print("%s [thread %d]: Processed %d of %d items in thread batch." %
                  (datetime.now(), thread_index, counter, num_images_in_thread))
            sys.stdout.flush()

        writer.close()
        print("%s [thread %d]: Wrote %d image-caption pairs to %s" %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print("%s [thread %d]: Wrote %d image-caption pairs to %d shards." %
        (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()

def process_dataset(name, images, vocab, num_shards):
    """Processes a complete data set and saves it as a TFRecord.

    Args:
    name: Unique identifier specifying the dataset.
    images: List of ImageMetadata.
    vocab: A Vocabulary object.
    num_shards: Integer number of shards for the output files.
    """
    # Break up each image into a separate entity for each caption.
    images = [ImageMetadata(image[0], image[1], [caption])
            for image in images for caption in image[2]]

    # Shuffle the ordering of images. Make the randomization repeatable.
    random.seed(12345)
    random.shuffle(images)

    # Break the images into num_threads batches. Batch i is defined as
    # images[ranges[i][0]:ranges[i][1]].
    num_threads = min(num_shards, FLAGS.num_threads)
    spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a utility for decoding JPEG images to run sanity checks.
    decoder = ImageDecoder()

    # Launch a thread for each batch.
    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, images, decoder, vocab, num_shards)
        t = threading.Thread(target=process_image_files, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print("%s: Finished processing all %d image-caption pairs in data set '%s'." %
        (datetime.now(), len(images), name))

def main(argv):
    image_metadata = []
    with io.open('/Users/KYD/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/img_metadata_train.json') as f:
        image_metadata = json.loads(f.read())

    word2id = dict()
    with io.open('/Users/KYD/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/word_to_index_dict.json') as f:
        word2id = json.loads(f.read())

    process_dataset('train', image_metadata, Vocabulary(word2id, len(word2id)), FLAGS.train_shards)

if __name__ == "__main__":
  tf.app.run()