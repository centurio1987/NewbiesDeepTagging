import tensorflow as tf
from im2txt.ops import image_processing
from models import make_skip_gram_for_image
from models import make_batch_input
from models import make_image_embeddings
from models import make_image_embeddings_cnn
import io
import json

'''
1. TFRecord 불러오기
2.
'''

tf.flags.DEFINE_string('input_file_pattern', '/Users/KYD/Downloads/tfrecord/train*','input file pattern')
tf.flags.DEFINE_integer('values_per_shard', 8, 'values_per_shard')
tf.flags.DEFINE_integer('input_queue_capacity_factor', 16, 'input_queue_capacity_factor')
tf.flags.DEFINE_integer('batch_size', 32, 'batch_size')
tf.flags.DEFINE_integer('image_height', 299, 'image_height')
tf.flags.DEFINE_integer('image_width', 299, 'image_width')
tf.flags.DEFINE_integer('resize_height', 346, 'resize_height')
tf.flags.DEFINE_integer('resize_width', 346, 'resize_width')
tf.flags.DEFINE_string('image_format', 'jpeg', 'image_format')
tf.flags.DEFINE_integer('num_preprocess_threads', 4, 'num_preprocess_threads')

class InputDataBuilder:
    def __init__(self):
        self.reader = tf.TFRecordReader()
        self.batch_contexts = None
        self.batch_images = None
        self.image_embeddings = None

        word2id = dict()
        with io.open('/Users/KYD/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/word_to_index_dict.json') as f:
            word2id = json.loads(f.read())

        self.voc_size = len(word2id)

    def prefetch_tfrecords(self):
        data_files = tf.gfile.Glob(tf.flags.FLAGS.input_file_pattern)

        if not data_files:
            tf.logging.fatal("Found no input files matching %s", tf.flags.FLAGS.input_file_pattern)
        else:
            tf.logging.info("Prefetching values from %d files matching %s",
                            len(data_files), tf.flags.FLAGS.input_file_pattern)

        filename_queue = tf.train.string_input_producer(
            data_files, shuffle=True, capacity=16, name='filename_queue')
        min_queue_examples = tf.flags.FLAGS.values_per_shard * tf.flags.FLAGS.input_queue_capacity_factor
        capacity = min_queue_examples + 100 * tf.flags.FLAGS.batch_size
        values_queue = tf.RandomShuffleQueue(
            capacity=capacity,
            min_after_dequeue=min_queue_examples,
            dtypes=[tf.string],
            name="random_" + 'input_queue')

        enqueue_ops = []
        num_reader_threads = 1
        for _ in range(num_reader_threads):
            _, value = self.reader.read(filename_queue)
            enqueue_ops.append(values_queue.enqueue([value]))
        tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(
            values_queue, enqueue_ops))
        tf.summary.scalar(
            "queue/%s/fraction_of_%d_full" % (values_queue.name, capacity),
            tf.cast(values_queue.size(), tf.float32) * (1. / capacity))

        return values_queue

    def build_input(self):
        #TFRecord로부터 이미지와 캡션 불러오기
        input_queue = self.prefetch_tfrecords()

        images_captions_pairs = []
        for thread_id in range(tf.flags.FLAGS.num_preprocess_threads):
            context, sequence = tf.parse_single_sequence_example(
                input_queue.dequeue(),
                context_features={
                    'image/data': tf.FixedLenFeature([], dtype=tf.string)
                },
                sequence_features={
                    'image/caption_ids': tf.FixedLenSequenceFeature([], dtype=tf.int64),
                })

            encoded_image = context['image/data']
            #captions = sequence['image/caption_ids']
            captions = sequence.values()

            # 이미지 디코딩 하기
            decoded_image = image_processing.process_image(encoded_image,
                                                           is_training=True,
                                                           thread_id=thread_id,
                                                           height=tf.flags.FLAGS.image_height,
                                                           width=tf.flags.FLAGS.image_width,
                                                           resize_height=tf.flags.FLAGS.resize_height,
                                                           resize_width=tf.flags.FLAGS.resize_width,
                                                           image_format=tf.flags.FLAGS.image_format)
            images_captions_pairs.append([decoded_image, captions])

        skip_gram_pairs = []
        for image, captions in images_captions_pairs:
            for word in captions:
                skip_gram_pairs.append((image, word))

        #skip_gram_pairs = make_skip_gram_for_image.make_skip_gram_for_image(images_captions_pairs)

        # 배치 입력 만들기
        self.batch_images, self.batch_contexts = make_batch_input.make_batch_input(skip_gram_pairs)

        self.image_embeddings = make_image_embeddings_cnn.make_image_embeddings_cnn(self.batch_images, self.voc_size)




