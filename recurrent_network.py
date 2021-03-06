from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=True)


# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 28
timesteps = 28 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 

# tf Graph input
# 注意：X 多了一個維度，timesteps
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
# 注意： weights 變得很簡潔
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)
    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)

    # 白話：把 timesteps 拿來當index，使從 3-dim 變成 2-dim
    # 此動作稱為 unstack，讓 shape 改變
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    # forget_bias=1.0 和上課影片的值相同
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # 反正就照做，Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    # 取 outputs 的最後一筆來算
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

# LSTM 的 logits 要傳好擠幾個參數，不像 cnn 只要 X，其他都從 global 拿就好
logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# 後面都一樣了
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # 小心：記得把 mnist 拿到的 raw data 轉成 (128, 28, 28)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # 就可以拿進去 train，Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    # 小心：做 testing 前，也要記得把 mnist 拿到的 raw data 轉成 (-1, 28, 28)
    # 因為 testing 不像 train_op 一樣有 batch_size，所以設為 -1 使自定義就好
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
