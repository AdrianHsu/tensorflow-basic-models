"""DNNRegressor with custom input_fn for Housing dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO) # 好用

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"


# 可以 process any of the DataFrames you've imported: 
# 有三種：training_set, test_set, and prediction_set.
def get_input_fn(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y=pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)


def main(unused_argv):
  # Load datasets
  training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS) # 把第一row的feature(header)拿掉
# training_set["medv"].values 就可以拿到 y 值

  test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,
                         skiprows=1, names=COLUMNS)

  # Set of 6 examples for which to predict median house values
  prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)

  # Feature cols
# Next, create a "list" of FeatureColumns for the input data, which formally 
# specify the set of features to use for training. 定義清楚train 用到的features
# Because all features in the housing data set contain continuous values, 
# you can create their FeatureColumn using .numeric_column()
# 一列一列column 根據 feature 名字取出來
  feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

  # 所以 feature columns 如何用？看進階教學（有教如何處理 categorical data）

  # print(feature_cols)
  # Build 2 layer fully connected DNN with 10, 10 units respectively.
  regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                        hidden_units=[10, 10], #here, two hidden layers with 10 nodes each
                                        model_dir="./tmp/boston_model")

  # Train
  regressor.train(input_fn=get_input_fn(training_set), steps=5000)

  # Evaluate loss over one epoch of test_set.
  ev = regressor.evaluate(
      input_fn=get_input_fn(test_set, num_epochs=1, shuffle=False))
  loss_score = ev["loss"]
  print("Loss: {0:f}".format(loss_score))

  # Print out predictions over a slice of prediction_set.
  y = regressor.predict(
      input_fn=get_input_fn(prediction_set, num_epochs=1, shuffle=False))
  # predict() returns an iterator of dicts; convert to a list and print
  # predictions
  predictions = list(p["predictions"] for p in itertools.islice(y, 6))
  print("Predictions: {}".format(str(predictions)))

if __name__ == "__main__":
  tf.app.run()