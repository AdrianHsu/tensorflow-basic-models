from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

import numpy as np
import tensorflow as tf
import argparse
import sys
tf.logging.set_verbosity(tf.logging.INFO) # 哇這真的好用～～什麼都印
# ref: Tensorflow r1.0 文档翻译，使用tf.contrib.learn记录和监视的基本知识

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main(_):

  # If the training and test sets aren't stored locally, download them.
  if not os.path.exists(IRIS_TRAINING):
    raw = urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, "wb") as f:
      f.write(raw)

  if not os.path.exists(IRIS_TEST):
    raw = urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, "wb") as f:
      f.write(raw)

  # Load datasets.
  # 可以善用 load_csv_with_header

  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

  # Specify that all features have real-value data
  feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]


  # Build 3 layer DNN with 10, 20, 10 units respectively.
  # 也有 DNNRegressor() 可以用
  classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10], # 三層，分別有 10, 20, 10個node
                                          n_classes=3,
                                          model_dir="./tmp/iris_model", # config 是我多加的
                                          config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))
  # 每次运行代码时，在/tmp/iris_model目录下的任何的数据都会被加载，
  # 并且模型训练将会在上次停止的位置继续进行
  # （例如，连续运行两次2000步fit操作的脚本将在训练期间执行4000步操作）
  # 如果想要从头开始模型训练，那么需要在执行训练前删除/tmp/iris_model目录。

  # Define the training inputs
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(training_set.data)},
      y=np.array(training_set.target),
      num_epochs=None,
      shuffle=True)

  # Train model.
  classifier.train(input_fn=train_input_fn, steps=2000)

  # Define the test inputs
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(test_set.data)},
      y=np.array(test_set.target),
      num_epochs=1,
      shuffle=False)

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score)) #   Test Accuracy: 0.966667
  # 以上主程式結束

  # 以下：遇到新的 data...
  # Classify two new flower samples.
  new_samples = np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": new_samples},
      num_epochs=1,
      shuffle=False)

  predictions = list(classifier.predict(input_fn=predict_input_fn))
  predicted_classes = [p["classes"] for p in predictions]

  print(
      "New Samples, Class Predictions:    {}\n"
      .format(predicted_classes))

# 結果：
# New Samples, Class Predictions:    [1 2]
# New Samples, Class Predictions:    [array([b'1'], dtype=object), array([b'2'], dtype=object)]

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # 這種 寫好的 DNNClassifier 不適用 log，直接用 SummarySaverHook
  
  # parser.add_argument( 
  #     '--log_dir',
  #     type=str,
  #     default=os.path.join(os.getenv('TEST_TMPDIR', './tmp'),
  #                          'logs/summaries'),
  #     help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
