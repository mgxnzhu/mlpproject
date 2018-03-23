# coding: utf-8

import numpy as np
import tensorflow as tf

DATA_DIR = "../data/"

n_input = 28
n_classes = 2
learning_rate = 0.0001
batch_size = 100
display_step = 25

def slicedata(data, dataset):
    trainset = data[dataset]
    features = np.float32(trainset[:,1:-2])
    labels_int = np.int32(trainset[:,-1])
    labels = np.zeros((labels_int.shape[0], n_classes))
    labels[range(labels_int.shape[0]), labels_int] = 1
    return features, labels

import argparse

parser = argparse.ArgumentParser(description='Set the evaluation dataset')
parser.add_argument('--evaluation', nargs="?", type=str, default='valid',
    help='"valid" or "test"')
parser.add_argument('--steps', nargs="?", type=int, default=500,
    help='Number of Trainning steps')
args = parser.parse_args()

evalset = args.evaluation
num_steps = args.steps

with np.load(DATA_DIR+"ccdataset.npz") as data:
    features, labels = slicedata(data, 'train')
    features_valid, labels_valid = slicedata(data, evalset)

# Assume that each row of `features` corresponds to the same row as `labels`.
assert features.shape[0] == labels.shape[0]

features_placeholder = tf.placeholder(tf.float32, [None, n_input])
labels_placeholder = tf.placeholder(tf.float32, [None, n_classes])

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
dataset = dataset.batch(batch_size)
iterator = dataset.make_initializable_iterator()

X, Y = iterator.get_next()

from My_model import modelname, Baseline_model

logits_train = Baseline_model(X, n_classes, reuse=False, is_training=True)
logits_test = Baseline_model(X, n_classes, reuse=True, is_training=False)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits_train, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

def modeleval(logits, Y):
    predicted = tf.argmax(logits, 1)
    actual = tf.argmax(Y, 1)

    tp = tf.count_nonzero(predicted * actual)
    tn = tf.count_nonzero((predicted - 1) * (actual - 1))
    fp = tf.count_nonzero(predicted * (actual - 1))
    fn = tf.count_nonzero((predicted - 1) * actual)

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fmeasure = (2 * precision * recall) / (precision + recall + 1e-8)

    return accuracy, fmeasure

accuracy, fmeasure = modeleval(logits_test, Y)

_features_valid = tf.placeholder(tf.float32, [None, n_input])
_labels_valid = tf.placeholder(tf.float32, [None, n_classes])

logits_val = Baseline_model(_features_valid, n_classes, reuse=True, is_training=True)
logits_val_test = Baseline_model(_features_valid, n_classes, reuse=True, is_training=False)
loss_val_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits_val, labels=_labels_valid))
acc_val_op, f1_val_op = modeleval(logits_val_test, _labels_valid)

stats = []

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

sess.run(iterator.initializer, feed_dict={features_placeholder: features,                                           labels_placeholder: labels})

for step in range(1, num_steps + 1):
    try:
        sess.run(train_op)
    except tf.errors.OutOfRangeError:
        sess.run(iterator.initializer, feed_dict={features_placeholder: features, labels_placeholder: labels})
        sess.run(train_op)

    loss, acc, f1 = sess.run([loss_op, accuracy, fmeasure])

    loss_val, acc_val, f1_val = sess.run([loss_val_op, acc_val_op, f1_val_op], feed_dict={_features_valid: features_valid, _labels_valid: labels_valid})

    stats.append([loss, acc, f1, loss_val, acc_val, f1_val])

    if step % display_step == 0 or step == 1:
        print("Step " + str(step) + ", Train Loss= " +               "{:.4f}".format(loss) + ", Train Acc= " +               "{:.3f}".format(acc)+ ", Train F1= " +               "{:.3f}".format(f1) + ", Valid Loss= " +               "{:.4f}".format(loss_val) + ", Valid Acc= " +               "{:.3f}".format(acc_val)+ ", Valid F1= " +               "{:.3f}".format(f1_val))
sess.close()

stats = np.array(stats)
np.save(modelname+".npy", stats)
