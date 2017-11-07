import argparse
import os
import sys

import tensorflow as tf

def train():
  sess = tf.InteractiveSession()

  # Model parameters
  x = tf.placeholder(tf.float32, name='x-input')
  y = tf.placeholder(tf.float32, name='y-input')

  W = tf.Variable([.3], dtype=tf.float32, name='weight')
  b = tf.Variable([-.3], dtype=tf.float32, name='bias')

  linear_model = W*x + b
  tf.summary.histogram('linear_model', linear_model)
  # loss
  loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
  # optimizer
  tf.summary.scalar('loss', loss);
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train_step = optimizer.minimize(loss)

  # training data
  x_train = [1, 2, 3, 4]
  y_train = [0, -1, -2, -3]
  # training loop
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
  
  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    return {x: x_train, y: y_train}

  init = tf.global_variables_initializer()
  sess.run(init) 

  for i in range(1000):
    # sess.run([merged, train], {x: x_train, y: y_train})
    if i % 100 == 0:
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      summary, _ = sess.run([merged, train_step],
                            feed_dict=feed_dict(True),
                            options=run_options,
                            run_metadata=run_metadata)
      train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
      train_writer.add_summary(summary, i)
    else:
      summary, _ = sess.run([merged, train], feed_dict=feed_dict(True))
      train_writer.add_summary(summary, i)

  # evaluate training accuracy
  curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
  print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', './tmp'),
                           'logs/lr_with_summaries'),
      help='Summaries log directory')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
