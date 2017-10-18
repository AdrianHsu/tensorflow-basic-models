from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=False)

import tensorflow as tf

# 參數：其中 num_steps 就是 epoch 數
# batch_size：一次吃 128 張圖
# num_steps is different from num_epochs
# An epoch usually means "one iteration over all of the training data". 
# if you have 20,000 images and a batch size of 100 
# then the epoch should contain 20,000 / 100 = 200 steps. 
learning_rate = 0.1
num_steps = 1000
batch_size = 128
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 第一層 hidden layer 的 neurons 數量
n_hidden_2 = 256
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)


def neural_net(x_dict):
    # TF Estimator input is a dict, in case of multiple inputs
    x = x_dict['images']

    layer_1 = tf.layers.dense(x, n_hidden_1)
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer


# Define the model function (following "TF Estimator Template")
# 利用 Estimator 呼叫，此 3 個參數都不用傳入
def model_fn(features, labels, mode):
    # 1. 利用寫好的 NN，建造 logits 
    logits = neural_net(features)

    # 最後一層 prediction
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)

    # If prediction mode, early return
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

    # 和 raw 版一樣，Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,
                                  global_step=tf.train.get_global_step())

    # （ raw 版沒有）Evaluate the accuracy of the model
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    # TF Estimators requires to return a EstimatorSpec, that specify
    # the different ops for training, evaluating, ...
    # 利用 EstimatorSpec 講清楚這個 model 的 loss function, optimizer, accuracy 如何定義
    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

# Estimator 評估器：進階 API，利用 model_fn 建構 model
model = tf.estimator.Estimator(model_fn)

# Define the input function for training
# 用來 train 的參數： x, y, batch_size, num_epochs...etc
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)

# 開始 Train Model，傳入 input function
model.train(input_fn, steps=num_steps)

# 訓練完畢，評估 Model 好壞
# Define the input function
# 用來 test 的參數： x, y, batch_size, num_epochs...etc
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)

# Use the Estimator 'evaluate' method
e = model.evaluate(input_fn)

print("Testing Accuracy:", e['accuracy'])
