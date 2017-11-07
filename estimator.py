# NumPy is often used to load, manipulate and preprocess data.
import numpy as np
import tensorflow as tf

# 声明特征集。在这个案例中我们只有一个特征。
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

# 定义一个学习器，此处是线性回归
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns, model_dir='./model_es')

# 定义数据集
# Tensorflow 提供了很多方法来读取与设置数据集。
# 这里使用 .numpy_input_fn()，括号中必须告诉这个函数，
# 要多少批数据（num_epochs), 以及每批数据的大小（batch_size)
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)


# 对estimator调用train()方法，然后传入训练的数据集，以及迭代的次数
# 以前的estimator 是用 fit() 但效果一樣
estimator.train(input_fn=input_fn, steps=1000)


# 评估模型
# 这里直接使用训练数据来评估。
# 运行以下代码，可以根据输出的损失loss来评估模型的fitting程度。
train_metrics = estimator.evaluate(input_fn=train_input_fn)
# 使用测试机或验证集来评估模型避免overfitting。
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)

print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)

# 印出的結果
# train metrics: {'average_loss': 5.4413022e-07, 'loss': 2.1765209e-06, 'global_step': 1000}
# eval metrics: {'average_loss': 0.0025950226, 'loss': 0.01038009, 'global_step': 1000}

# WARNING:tensorflow:Using temporary folder as model directory: /var/folders/5l/3vwlkr_n64b2rfjz6585jwkm0000gn/T/tmpxa23qg5n
# 如何解決？ 補上 model_dir 即可。
# estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns, model_dir='./model')
