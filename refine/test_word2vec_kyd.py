import json
import io
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# skip-gram 모델 만들기
# batch-set 만들기
# 임베딩테이블 만들기
# 학습하기

def generate_batch(size):
    acc_count = 0
    while len(skip_gram_pairs) < acc_count:
        x_inputs = []
        y_inputs = []
        for i in range(acc_count, size):
            x_inputs.append(skip_gram_pairs[i][0])
            y_inputs.append(skip_gram_pairs[i][1])

        acc_count += size
        yield x_inputs, y_inputs

dic = dict()
rdic = []
window_size = 2

with io.open('/Users/KYD/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/word_to_index_dict.json', 'r') as f:
    dic = json.loads(f.read())

with io.open('/Users/KYD/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/index_to_word_list.json', 'r') as f:
    rdic = json.loads(f.read())

index_data = []
with io.open('/Users/KYD/Dropbox/논문/MSCOCO/captions_train-val2014/annotations/trimmed_tokenized_caption.json', 'r') as f:
    trimmed_tokenized_caption = json.loads(f.read())

    for caption in trimmed_tokenized_caption:
        templist = []
        for token in caption:
            templist.append(dic[token])

        index_data.append(templist)

#skip-gram pair 만들기
skip_gram_pairs = []
for sequence in index_data:
    for target_index in range(len(sequence)):
        for window_index in range(0 - window_size, window_size+1):
            context_index = target_index + window_index
            if 0 <= context_index < len(sequence) and target_index is not context_index:
                skip_gram_pairs.append([sequence[target_index], sequence[context_index]])

#constructing word2vec neural net graph

batch_size = 10
step_size = 100
embedding_size = 2
neg_sample_size = 10
voc_size = len(dic)
X = tf.placeholder(shape=[batch_size], dtype=tf.int32)
Y = tf.placeholder(shape=[1, batch_size], dtype=tf.int32)

with tf.device('/cpu:0'):
    embedding_table = tf.Variable(tf.random_uniform(shape=[voc_size, embedding_size], minval=-1.0, maxval=1.0))
    embed = tf.nn.embedding_lookup(embedding_table, X)

nce_weight = tf.Variable(tf.random_uniform(shape=[voc_size, embedding_size], minval=-1.0, maxval=1.0))
nce_bias = tf.Variable(tf.zeros(shape=[voc_size]))

loss = tf.reduce_mean(tf.nn.nce_loss(nce_weight,nce_bias,embed,Y,neg_sample_size,voc_size))

train = tf.train.AdamOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for step in range(step_size):
        for x_inputs, y_inputs in generate_batch(batch_size):
            __, loss_val = sess.run([train, loss], feed_dict={X:x_inputs, Y:y_inputs})

            if step % 10 == 0:
                print("Loss at ", step, loss_val)  # Report the loss

    trained_embeddings = embedding_table.eval()

# Show word2vec if dim is 2
if trained_embeddings.shape[1] == 2:
    labels = rdic[:10] # Show top 10 words
    for i, label in enumerate(labels):
        x, y = trained_embeddings[i,:]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2),
            textcoords='offset points', ha='right', va='bottom')
    plt.savefig("word2vec.png")
