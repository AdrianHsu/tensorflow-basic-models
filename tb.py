import tensorflow as tf

with tf.name_scope('hidden') as scope:
  a = tf.constant(5, name='alpha')
  W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
  b = tf.Variable(tf.zeros([1]), name='biases')

The better your name scopes, the better your visualization.

(expended view)
1. data dependencies 
  Data dependencies show the flow of tensors between two ops and are shown 
  as solid arrows.

2. control dependencies. 
  control dependencies use dotted lines. 
  