from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=True)

import tensorflow as tf


learning_rate = 0.1
num_steps = 500
batch_size = 128
display_step = 100


n_hidden_1 = 256 
n_hidden_2 = 256 
num_input = 784 # 28*28, flatten 把像素拉直
num_classes = 10 


X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])


weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
    # 注意：接 output 也有 weights 要考慮
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def neural_net(x):
    # Wx + b, Hidden layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model, 最後過一層 softmax
logits = neural_net(X)
prediction = tf.nn.softmax(logits) 

# 以下都一樣
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
# correct_pred 比對 prediction 和 Y 的值
# reduce_mean() 求平均
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# Initialize the variables (i.e. assign their default value)
# 呼叫 global_variables_initializer() 幫變數賦值
init = tf.global_variables_initializer()

# 建立一個 session
with tf.Session() as sess:

    # Run the initializer, 在 session 的一開始先幫變數賦值
    sess.run(init) 

# 讀入 500 training examples in each training iteration. 
# We then run the train_op operation, using feed_dict to replace 
# the placeholder tensors X and Y with the training examples.

    for step in range(1, num_steps+1): # 1 to 501
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        
        # 反向傳播：Run optimization op (backprop)
        # feed_dict 吃的是 batch_x, batch_y
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        if step % display_step == 0 or step == 1:
            # 印出 batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    # 印出用 testing data 的 accuracy
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: mnist.test.images,
                                      Y: mnist.test.labels}))
