from __future__ import division, print_function, absolute_import

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=False)

import tensorflow as tf

learning_rate = 0.001
num_steps = 2000
batch_size = 128

# Network Parameters
num_input = 784
num_classes = 10 
dropout = 0.75 # Dropout, probability to keep units
# dropout 是為了解決 overfitting，但也會拖慢速度

# 利用 keep_prob = 0.75 決定 each connection between layers 
# will only be used with this probability when training.


def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    
    # Define a scope for reusing the variables
    # 為何要 reuse，因為 testing phase 會測好幾次，這模型不能改變
    with tf.variable_scope('ConvNet', reuse=reuse):

        x = x_dict['images']
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector
        fc1 = tf.contrib.layers.flatten(conv2)
        # 加一層 hidden layer， 接到 size=1024 的 hidden layer 1
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout 
        # (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer，是 hidden layer 1 接 output，class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out

# Define the model function (following TF Estimator Template)
# 和 nn.py 一樣的 3 個參數
def model_fn(features, labels, mode):
    # Build the neural network
    # 因為 train / test 要用不同 dropout，所以建兩個 conv_net 但是 share same weights
    # Because Dropout have different behavior at training and prediction time, we
    # need to create 2 distinct computation graphs that still share the same weights.
    logits_train = conv_net(features, num_classes, dropout, reuse=False,
                            is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True,
                           is_training=False)


    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # 看過了，去看 nn.py
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

# Build the Estimator
model = tf.estimator.Estimator(model_fn)

# 以下都跟 nn.py 一樣，回去看註解
# Define the input function for training
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# Train the Model
model.train(input_fn, steps=num_steps)

# Evaluate the Model
# Define the input function for evaluating
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])
