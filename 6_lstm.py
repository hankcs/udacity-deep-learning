# coding: utf-8

# 

# Deep Learning
# =============
# 
# Assignment 6
# ------------
# 
# After training a skip-gram model in `5_word2vec.ipynb`, the goal of this notebook is to train a LSTM character model over [Text8](http://mattmahoney.net/dc/textdata) data.

# In[ ]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve

# In[ ]:

url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


filename = maybe_download('text8.zip', 31344016)


# In[ ]:

def read_data(filename):
    f = zipfile.ZipFile(filename)
    for name in f.namelist():
        return tf.compat.as_str(f.read(name))
    f.close()


text = read_data(filename)
print('Data size %d' % len(text))

# Create a small validation set.

# In[ ]:

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])

# Utility functions to map characters to vocabulary IDs and back.

# In[ ]:

vocabulary_size = len(string.ascii_lowercase) + 1  # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])


def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
        return 0


def id2char(dictid):
    if dictid > 0:
        return chr(dictid + first_letter - 1)
    else:
        return ' '


print(char2id('a'), char2id('z'), char2id(' '), char2id('ï'))
print(id2char(1), id2char(26), id2char(0))

# Function to generate a training batch for the LSTM model.

# In[ ]:

batch_size = 64
num_unrollings = 10


class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b, char2id(self._text[self._cursor[b]])] = 1.0
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(c) for c in np.argmax(probabilities, 1)]


def batches2string(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s


train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

print(batches2string(train_batches.next()))
print(batches2string(train_batches.next()))
print(batches2string(valid_batches.next()))
print(batches2string(valid_batches.next()))


# In[ ]:

def logprob(predictions, labels):
    """Log-probability of the true labels in a predicted batch."""
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]


def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    """
    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
        s += distribution[i]
        if s >= r:
            return i
    return len(distribution) - 1


def sample(prediction):
    """Turn a (column) prediction into 1-hot encoded samples."""
    p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p


def random_distribution():
    """Generate a random column of probabilities."""
    b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
    return b / np.sum(b, 1)[:, None]


# Simple LSTM Model.

# In[ ]:

num_nodes = 64

graph = tf.Graph()
with graph.as_default():
    # Parameters:
    # Input gate: input, previous output, and bias.
    ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ib = tf.Variable(tf.zeros([1, num_nodes]))
    # Forget gate: input, previous output, and bias.
    fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    fb = tf.Variable(tf.zeros([1, num_nodes]))
    # Memory cell: input, state and bias.
    cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([1, num_nodes]))
    # Output gate: input, previous output, and bias.
    ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ob = tf.Variable(tf.zeros([1, num_nodes]))
    # Variables saving state across unrollings.
    saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    # Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([vocabulary_size]))


    # Definition of the cell computation.
    def lstm_cell(i, o, state):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""
        input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
        forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
        update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
        state = forget_gate * state + input_gate * tf.tanh(update)
        output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
        return output_gate * tf.tanh(state), state


    # Input data.
    train_data = list()
    for _ in range(num_unrollings + 1):
        train_data.append(
            tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size]))
    train_inputs = train_data[:num_unrollings]
    train_labels = train_data[1:]  # labels are inputs shifted by one time step.

    # Unrolled LSTM loop.
    outputs = list()
    output = saved_output
    state = saved_state
    for i in train_inputs:
        output, state = lstm_cell(i, output, state)
        outputs.append(output)

    # State saving across unrollings.
    with tf.control_dependencies([saved_output.assign(output),
                                  saved_state.assign(state)]):
        # Classifier.
        logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits, tf.concat(0, train_labels)))

    # Optimizer.
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        10.0, global_step, 5000, 0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(
        zip(gradients, v), global_step=global_step)

    # Predictions.
    train_prediction = tf.nn.softmax(logits)

    # Sampling and validation eval: batch 1, no unrolling.
    sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
    saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
    saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
    reset_sample_state = tf.group(
        saved_sample_output.assign(tf.zeros([1, num_nodes])),
        saved_sample_state.assign(tf.zeros([1, num_nodes])))
    sample_output, sample_state = lstm_cell(
        sample_input, saved_sample_output, saved_sample_state)
    with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                  saved_sample_state.assign(sample_state)]):
        sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

# In[ ]:

num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    mean_loss = 0
    for step in range(num_steps):
        batches = train_batches.next()
        feed_dict = dict()
        for i in range(num_unrollings + 1):
            feed_dict[train_data[i]] = batches[i]
        _, l, predictions, lr = session.run(
            [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
        mean_loss += l
        if step % summary_frequency == 0:
            if step > 0:
                mean_loss = mean_loss / summary_frequency
            # The mean loss is an estimate of the loss over the last few batches.
            print(
                'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
            mean_loss = 0
            labels = np.concatenate(list(batches)[1:])
            print('Minibatch perplexity: %.2f' % float(
                np.exp(logprob(predictions, labels))))
            if step % (summary_frequency * 10) == 0:
                # Generate some samples.
                print('=' * 80)
                for _ in range(5):
                    feed = sample(random_distribution())
                    sentence = characters(feed)[0]
                    reset_sample_state.run()
                    for _ in range(79):
                        prediction = sample_prediction.eval({sample_input: feed})
                        feed = sample(prediction)
                        sentence += characters(feed)[0]
                    print(sentence)
                print('=' * 80)
            # Measure validation set perplexity.
            reset_sample_state.run()
            valid_logprob = 0
            for _ in range(valid_size):
                b = valid_batches.next()
                predictions = sample_prediction.eval({sample_input: b[0]})
                valid_logprob = valid_logprob + logprob(predictions, b[1])
            print('Validation set perplexity: %.2f' % float(np.exp(
                valid_logprob / valid_size)))

# ---
# Problem 1
# ---------
#
# You might have noticed that the definition of the LSTM cell involves 4 matrix multiplications with the input, and 4 matrix multiplications with the output. Simplify the expression by using a single matrix multiply for each, and variables that are 4 times larger.
#
# ---
num_nodes = 64

graph = tf.Graph()
with graph.as_default():
    # Parameters:
    # Input gate: input, previous output, and bias.
    ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ib = tf.Variable(tf.zeros([1, num_nodes]))
    # Forget gate: input, previous output, and bias.
    fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    fb = tf.Variable(tf.zeros([1, num_nodes]))
    # Memory cell: input, state and bias.
    cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([1, num_nodes]))
    # Output gate: input, previous output, and bias.
    ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ob = tf.Variable(tf.zeros([1, num_nodes]))
    # Concatenate parameters
    sx = tf.concat(1, [ix, fx, cx, ox])
    sm = tf.concat(1, [im, fm, cm, om])
    sb = tf.concat(1, [ib, fb, cb, ob])
    # Variables saving state across unrollings.
    saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    # Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([vocabulary_size]))


    # Definition of the cell computation.
    def lstm_cell(i, o, state):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""
        y = tf.matmul(i, sx) + tf.matmul(o, sm) + sb
        y_input, y_forget, update, y_output = tf.split(1, 4, y)
        input_gate = tf.sigmoid(y_input)
        forget_gate = tf.sigmoid(y_forget)
        output_gate = tf.sigmoid(y_output)
        state = forget_gate * state + input_gate * tf.tanh(update)
        return output_gate * tf.tanh(state), state


    # Input data.
    train_data = list()
    for _ in range(num_unrollings + 1):
        train_data.append(
            tf.placeholder(tf.float32, shape=[batch_size, vocabulary_size]))
    train_inputs = train_data[:num_unrollings]
    train_labels = train_data[1:]  # labels are inputs shifted by one time step.

    # Unrolled LSTM loop.
    outputs = list()
    output = saved_output
    state = saved_state
    for i in train_inputs:
        output, state = lstm_cell(i, output, state)
        outputs.append(output)

    # State saving across unrollings.
    with tf.control_dependencies([saved_output.assign(output),
                                  saved_state.assign(state)]):
        # Classifier.
        logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits, tf.concat(0, train_labels)))

    # Optimizer.
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        10.0, global_step, 5000, 0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(
        zip(gradients, v), global_step=global_step)

    # Predictions.
    train_prediction = tf.nn.softmax(logits)

    # Sampling and validation eval: batch 1, no unrolling.
    sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
    saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
    saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
    reset_sample_state = tf.group(
        saved_sample_output.assign(tf.zeros([1, num_nodes])),
        saved_sample_state.assign(tf.zeros([1, num_nodes])))
    sample_output, sample_state = lstm_cell(
        sample_input, saved_sample_output, saved_sample_state)
    with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                  saved_sample_state.assign(sample_state)]):
        sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    mean_loss = 0
    for step in range(num_steps):
        batches = train_batches.next()
        feed_dict = dict()
        for i in range(num_unrollings + 1):
            feed_dict[train_data[i]] = batches[i]
        _, l, predictions, lr = session.run(
            [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
        mean_loss += l
        if step % summary_frequency == 0:
            if step > 0:
                mean_loss = mean_loss / summary_frequency
            # The mean loss is an estimate of the loss over the last few batches.
            print(
                'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
            mean_loss = 0
            labels = np.concatenate(list(batches)[1:])
            print('Minibatch perplexity: %.2f' % float(
                np.exp(logprob(predictions, labels))))
            if step % (summary_frequency * 10) == 0:
                # Generate some samples.
                print('=' * 80)
                for _ in range(5):
                    feed = sample(random_distribution())
                    sentence = characters(feed)[0]
                    reset_sample_state.run()
                    for _ in range(79):
                        prediction = sample_prediction.eval({sample_input: feed})
                        feed = sample(prediction)
                        sentence += characters(feed)[0]
                    print(sentence)
                print('=' * 80)
            # Measure validation set perplexity.
            reset_sample_state.run()
            valid_logprob = 0
            for _ in range(valid_size):
                b = valid_batches.next()
                predictions = sample_prediction.eval({sample_input: b[0]})
                valid_logprob = valid_logprob + logprob(predictions, b[1])
            print('Validation set perplexity: %.2f' % float(np.exp(
                valid_logprob / valid_size)))
# ---
# Problem 2
# ---------
#
# We want to train a LSTM over bigrams, that is pairs of consecutive characters like 'ab' instead of single characters like 'a'. Since the number of possible bigrams is large, feeding them directly to the LSTM using 1-hot encodings will lead to a very sparse representation that is very wasteful computationally.
#
# a- Introduce an embedding lookup on the inputs, and feed the embeddings to the LSTM cell instead of the inputs themselves.
#
# b- Write a bigram-based LSTM, modeled on the character LSTM above.
#
# c- Introduce Dropout. For best practices on how to use Dropout in LSTMs, refer to this [article](http://arxiv.org/abs/1409.2329).
#
# ---
bigram_vocabulary_size = vocabulary_size * vocabulary_size


class BigramBatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size_in_chars = len(text)
        self._text_size = self._text_size_in_chars // 2
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()

    def _next_batch(self):
        batch = np.zeros(shape=self._batch_size, dtype=np.int)
        for b in range(self._batch_size):
            char_idx = self._cursor[b] * 2
            ch1 = char2id(self._text[char_idx])
            if self._text_size_in_chars - 1 == char_idx:
                ch2 = 0
            else:
                ch2 = char2id(self._text[char_idx + 1])
            batch[b] = ch1 * vocabulary_size + ch2
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        batches = [self._last_batch]
        for step in range(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


def bi2str(encoding):
    return id2char(encoding // vocabulary_size) + id2char(encoding % vocabulary_size)


def bigrams(encodings):
    return [bi2str(e) for e in encodings]


def bibatches2string(batches):
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, bigrams(b))]
    return s


bi_onehot = np.zeros((bigram_vocabulary_size, bigram_vocabulary_size))
np.fill_diagonal(bi_onehot, 1)


def bi_one_hot(encodings):
    return [bi_onehot[e] for e in encodings]


train_batches = BigramBatchGenerator(train_text, 8, 8)
valid_batches = BigramBatchGenerator(valid_text, 1, 1)

print(bibatches2string(train_batches.next()))
print(bibatches2string(train_batches.next()))
print(bibatches2string(valid_batches.next()))
print(bibatches2string(valid_batches.next()))


def logprob(predictions, labels):
    """Log-probability of the true labels in a predicted batch."""
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]


def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    """
    r = random.uniform(0, 1)
    s = 0
    for i in range(len(distribution)):
        s += distribution[i]
        if s >= r:
            return i
    return len(distribution) - 1


def sample(prediction, size=vocabulary_size):
    """Turn a (column) prediction into 1-hot encoded samples."""
    p = np.zeros(shape=[1, size], dtype=np.float)
    p[0, sample_distribution(prediction[0])] = 1.0
    return p


def one_hot_voc(prediction, size=vocabulary_size):
    p = np.zeros(shape=[1, size], dtype=np.float)
    p[0, prediction[0]] = 1.0
    return p


def random_distribution(size=vocabulary_size):
    """Generate a random column of probabilities."""
    b = np.random.uniform(0.0, 1.0, size=[1, size])
    return b / np.sum(b, 1)[:, None]


num_nodes = 512
num_unrollings = 10
batch_size = 32
embedding_size = 128
graph = tf.Graph()
with graph.as_default():
    # input to all gates
    x = tf.Variable(tf.truncated_normal([embedding_size, num_nodes * 4], -0.1, 0.1), name='x')
    # memory of all gates
    m = tf.Variable(tf.truncated_normal([num_nodes, num_nodes * 4], -0.1, 0.1), name='m')
    # biases all gates
    biases = tf.Variable(tf.zeros([1, num_nodes * 4]))
    # Variables saving state across unrollings.
    saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    # Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([num_nodes, bigram_vocabulary_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([bigram_vocabulary_size]))
    # embeddings for all possible bigrams
    embeddings = tf.Variable(tf.random_uniform([bigram_vocabulary_size, embedding_size], -1.0, 1.0))
    # one hot encoding for labels in
    np_one_hot = np.zeros((bigram_vocabulary_size, bigram_vocabulary_size))
    np.fill_diagonal(np_one_hot, 1)
    bigram_one_hot = tf.constant(np.reshape(np_one_hot, -1), dtype=tf.float32,
                                 shape=[bigram_vocabulary_size, bigram_vocabulary_size])
    keep_prob = tf.placeholder(tf.float32)


    # Definition of the cell computation.
    def lstm_cell(i, o, state):
        i = tf.nn.dropout(i, keep_prob)
        mult = tf.matmul(i, x) + tf.matmul(o, m) + biases
        input_gate = tf.sigmoid(mult[:, :num_nodes])
        forget_gate = tf.sigmoid(mult[:, num_nodes:num_nodes * 2])
        update = mult[:, num_nodes * 3:num_nodes * 4]
        state = forget_gate * state + input_gate * tf.tanh(update)
        output_gate = tf.sigmoid(mult[:, num_nodes * 3:])
        output = tf.nn.dropout(output_gate * tf.tanh(state), keep_prob)
        return output, state


    # Input data. [num_unrollings, batch_size] -> one hot encoding removed, we send just bigram ids
    tf_train_data = tf.placeholder(tf.int32, shape=[num_unrollings + 1, batch_size])
    train_data = list()
    for i in tf.split(0, num_unrollings + 1, tf_train_data):
        train_data.append(tf.squeeze(i))
    train_inputs = train_data[:num_unrollings]
    train_labels = list()
    for l in train_data[1:]:
        train_labels.append(tf.gather(bigram_one_hot, l))

    # Unrolled LSTM loop.
    outputs = list()
    output = saved_output
    state = saved_state
    # python loop used: tensorflow does not support sequential operations yet
    for i in train_inputs:  # having a loop simulates having time
        # embed input bigrams -> [batch_size, embedding_size]
        output, state = lstm_cell(tf.nn.embedding_lookup(embeddings, i), output, state)
        outputs.append(output)

    # State saving across unrollings, control_dependencies makes sure that output and state are computed
    with tf.control_dependencies([saved_output.assign(output), saved_state.assign(state)]):
        logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,
                                                                      tf.concat(0, train_labels)
                                                                      ))
    # Optimizer.
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(10.0, global_step, 500, 0.9, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

    # here we predict the embedding
    # train_prediction = tf.argmax(tf.nn.softmax(logits), 1, name='train_prediction')
    train_prediction = tf.nn.softmax(logits)

    # Sampling and validation eval: batch 1, no unrolling.
    sample_input = tf.placeholder(tf.int32, shape=[1])
    saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
    saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
    reset_sample_state = tf.group(saved_sample_output.assign(tf.zeros([1, num_nodes])),
                                  saved_sample_state.assign(tf.zeros([1, num_nodes])))
    embed_sample_input = tf.nn.embedding_lookup(embeddings, sample_input)
    sample_output, sample_state = lstm_cell(embed_sample_input, saved_sample_output, saved_sample_state)

    with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                  saved_sample_state.assign(sample_state)]):
        sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))

num_steps = 4001
summary_frequency = 100
# initalize batch generators

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    train_batches = BigramBatchGenerator(train_text, batch_size, num_unrollings)
    valid_batches = BigramBatchGenerator(valid_text, 1, 1)
    mean_loss = 0
    for step in range(num_steps):
        batches = train_batches.next()
        _, l, lr, predictions = session.run([optimizer, loss, learning_rate, train_prediction],
                                            feed_dict={tf_train_data: batches, keep_prob: 0.6})
        mean_loss += l
        if step % summary_frequency == 0:
            if step > 0:
                mean_loss = mean_loss / summary_frequency
            # The mean loss is an estimate of the loss over the last few batches.
            print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
            mean_loss = 0
            labels = list(batches)[1:]
            labels = np.concatenate([bi_one_hot(l) for l in labels])
            print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, labels))))
            if step % (summary_frequency * 10) == 0:
                # Generate some samples.
                print('=' * 80)
                for _ in range(5):
                    feed = np.argmax(sample(random_distribution(bigram_vocabulary_size), bigram_vocabulary_size))
                    sentence = bi2str(feed)
                    reset_sample_state.run()
                    for _ in range(49):
                        prediction = sample_prediction.eval({sample_input: [feed], keep_prob: 1.0})
                        feed = np.argmax(sample(prediction, bigram_vocabulary_size))
                        sentence += bi2str(feed)
                    print(sentence)
                print('=' * 80)
            # Measure validation set perplexity.
            reset_sample_state.run()
            valid_logprob = 0
            for _ in range(valid_size):
                b = valid_batches.next()
                predictions = sample_prediction.eval({sample_input: b[0], keep_prob: 1.0})
                # print(predictions)
                valid_logprob = valid_logprob + logprob(predictions, one_hot_voc(b[1], bigram_vocabulary_size))
            print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))

# ---
# Problem 3
# ---------
# 
# (difficult!)
# 
# Write a sequence-to-sequence LSTM which mirrors all the words in a sentence. For example, if your input is:
# 
#     the quick brown fox
#     
# the model should attempt to output:
# 
#     eht kciuq nworb xof
#     
# Refer to the lecture on how to put together a sequence-to-sequence model, as well as [this article](http://arxiv.org/abs/1409.3215) for best practices.
# 
# ---
from tensorflow.models.rnn.translate import seq2seq_model
import math

batch_size = 64
num_unrollings = 19


class Seq2SeqBatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // num_unrollings
        self._cursor = [offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch(0)

    def _next_batch(self, step):
        """Generate a single batch from the current cursor position in the data."""
        batch = ''
        # print('text size', self._text_size)
        for b in range(self._num_unrollings):
            # print(self._cursor[step])
            self._cursor[step] %= self._text_size
            batch += self._text[self._cursor[step]]
            self._cursor[step] += 1
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in range(self._batch_size):
            batches.append(self._next_batch(step))
        self._last_batch = batches[-1]
        return batches


def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(c) for c in np.argmax(probabilities, 1)]


def ids(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [str(c) for c in np.argmax(probabilities, 1)]


def batches2id(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, ids(b))]
    return s


train_batches = Seq2SeqBatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = Seq2SeqBatchGenerator(valid_text, 1, num_unrollings)


def rev_id(forward):
    temp = forward.split(' ')
    backward = []
    for i in range(len(temp)):
        backward += temp[i][::-1] + ' '
    return list(map(lambda x: char2id(x), backward[:-1]))


batches = train_batches.next()
train_sets = []
batch_encs = list(map(lambda x: list(map(lambda y: char2id(y), list(x))), batches))
batch_decs = list(map(lambda x: rev_id(x), batches))
print('x=', ''.join([id2char(x) for x in batch_encs[0]]))
print('y=', ''.join([id2char(x) for x in batch_decs[0]]))


def create_model(forward_only):
    model = seq2seq_model.Seq2SeqModel(source_vocab_size=vocabulary_size,
                                       target_vocab_size=vocabulary_size,
                                       buckets=[(20, 20)],
                                       size=256,
                                       num_layers=4,
                                       max_gradient_norm=5.0,
                                       batch_size=batch_size,
                                       learning_rate=1.0,
                                       learning_rate_decay_factor=0.9,
                                       use_lstm=True,
                                       forward_only=forward_only)
    return model


with tf.Session() as sess:
    model = create_model(False)
    sess.run(tf.initialize_all_variables())
    num_steps = 30001

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    step_ckpt = 100
    valid_ckpt = 500

    for step in range(1, num_steps):
        model.batch_size = batch_size
        batches = train_batches.next()
        train_sets = []
        batch_encs = list(map(lambda x: list(map(lambda y: char2id(y), list(x))), batches))
        batch_decs = list(map(lambda x: rev_id(x), batches))
        for i in range(len(batch_encs)):
            train_sets.append((batch_encs[i], batch_decs[i]))

        # Get a batch and make a step.
        encoder_inputs, decoder_inputs, target_weights = model.get_batch([train_sets], 0)
        _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, 0, False)

        loss += step_loss / step_ckpt

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if step % step_ckpt == 0:
            # Print statistics for the previous epoch.
            perplexity = math.exp(loss) if loss < 300 else float('inf')
            print("global step %d learning rate %.4f perplexity "
                  "%.2f" % (model.global_step.eval(), model.learning_rate.eval(), perplexity))
            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                sess.run(model.learning_rate_decay_op)
            previous_losses.append(loss)

            loss = 0.0

            if step % valid_ckpt == 0:
                v_loss = 0.0

                model.batch_size = 1
                batches = ['the quick brown fox']
                test_sets = []
                batch_encs = list(map(lambda x: list(map(lambda y: char2id(y), list(x))), batches))
                # batch_decs = map(lambda x: rev_id(x), batches)
                test_sets.append((batch_encs[0], []))
                # Get a 1-element batch to feed the sentence to the model.
                encoder_inputs, decoder_inputs, target_weights = model.get_batch([test_sets], 0)
                # Get output logits for the sentence.
                _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, 0, True)

                # This is a greedy decoder - outputs are just argmaxes of output_logits.
                outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

                print('>>>>>>>>> ', batches[0], ' -> ', ''.join(map(lambda x: id2char(x), outputs)))

                for _ in range(valid_size):
                    model.batch_size = 1
                    v_batches = valid_batches.next()
                    valid_sets = []
                    v_batch_encs = list(map(lambda x: list(map(lambda y: char2id(y), list(x))), v_batches))
                    v_batch_decs = list(map(lambda x: rev_id(x), v_batches))
                    for i in range(len(v_batch_encs)):
                        valid_sets.append((v_batch_encs[i], v_batch_decs[i]))
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch([valid_sets], 0)
                    _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, 0, True)
                    v_loss += eval_loss / valid_size

                eval_ppx = math.exp(v_loss) if v_loss < 300 else float('inf')
                print("  valid eval:  perplexity %.2f" % (eval_ppx))

    # reuse variable -> subdivide into two boxes
    model.batch_size = 1  # We decode one sentence at a time.
    batches = ['the quick brown fox']
    test_sets = []
    batch_encs = list(map(lambda x: list(map(lambda y: char2id(y), list(x))), batches))
    # batch_decs = map(lambda x: rev_id(x), batches)
    test_sets.append((batch_encs[0], []))
    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch([test_sets], 0)
    # Get output logits for the sentence.
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, 0, True)
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
    print('## : ', outputs)
    # If there is an EOS symbol in outputs, cut them at that point.
    if char2id('!') in outputs:
        outputs = outputs[:outputs.index(char2id('!'))]

    print(batches[0], ' -> ', ''.join(map(lambda x: id2char(x), outputs)))
