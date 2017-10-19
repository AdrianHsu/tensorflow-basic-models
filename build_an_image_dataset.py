""" Build an Image Dataset in TensorFlow.

For this example, you need to make your own set of images (JPEG).
We will show 2 different ways to build that dataset:

- From a root folder, that will have a sub-folder containing images for each class
    ```
    ROOT_FOLDER
       |-------- SUBFOLDER (CLASS 0)
       |             |
       |             | ----- image1.jpg
       |             | ----- image2.jpg
       |             | ----- etc...
       |             
       |-------- SUBFOLDER (CLASS 1)
       |             |
       |             | ----- image1.jpg
       |             | ----- image2.jpg
       |             | ----- etc...
    ```

- From a plain text file, that will list all images with their class ID:
    ```
    /path/to/image/1.jpg CLASS_ID
    /path/to/image/2.jpg CLASS_ID
    /path/to/image/3.jpg CLASS_ID
    /path/to/image/4.jpg CLASS_ID
    etc...
    ```

Below, there are some parameters that you need to change (Marked 'CHANGE HERE'), 
such as the dataset path.


"""
from __future__ import print_function

import tensorflow as tf
import os

MODE = 'folder'
DATASET_PATH = './dat'

N_CLASSES = 2
IMG_HEIGHT = 256
IMG_WIDTH = 340
CHANNELS = 3

dataset_path = DATASET_PATH
mode = MODE
batch_size = 128

# 懶得用 function call，直接一行行跑吧
# def read_images(dataset_path, mode, batch_size):
imagepaths, labels = list(), list()
if mode == 'file':
    data = open(dataset_path, 'r').read().splitlines()
    for d in data:
        imagepaths.append(d.split(' ')[0])
        labels.append(int(d.split(' ')[1]))
elif mode == 'folder':
    label = 0
    classes = sorted(os.walk(dataset_path).__next__()[1])
#     print(classes) #['class0', 'class1']
    for c in classes:
        c_dir = os.path.join(dataset_path, c)
        walk = os.walk(c_dir).__next__()
#         print(walk) # if c == 0, if c == 1
        for sample in walk[2]:
            if sample.endswith('.jpg') or sample.endswith('.jpeg'):
                imagepaths.append(os.path.join(c_dir, sample))
                labels.append(label)

        label += 1
else:
    raise Exception("Unknown Mode.")
# print(labels) #[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
# print(imagepaths)    #['./dat/class0/image_0001.jpg', './dat/class0/image_0002.jpg', './dat/class0/image_0003.jpg', './dat/class0/image_0004.jpg', './dat/class0/image_0005.jpg', './dat/class1/image_0011.jpg', './dat/class1/image_0012.jpg', './dat/class1/image_0013.jpg', './dat/class1/image_0014.jpg', './dat/class1/image_0015.jpg']

# Convert to Tensor
imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
labels = tf.convert_to_tensor(labels, dtype=tf.int32)
# Build a TF Queue, shuffle data (AH:  NOT shuffle for me)
image, label = tf.train.slice_input_producer([imagepaths, labels],
                                             shuffle=False)

# Read images from disk
image = tf.read_file(image)
image = tf.image.decode_jpeg(image, channels=CHANNELS)

# Resize images to a common size
image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

# Normalize... Why 127.5?
# AH: the purpose of this normalization is to bring the values in range [-1.0 , 1.0]. 
# As values for a pixel in grayscale are in range [0,255], 
# we needed to divide by 127.5, to bring 255 to 2.0.
image = image * 1.0/127.5 - 1.0

# Create batches
X, Y = tf.train.batch([image, label], batch_size=batch_size,
                      capacity=batch_size * 8,
                      num_threads=4)

## YEAH! We finished the data processing part!
## let's test these data with basic CNN as we learned before

# 底下的 code 我就沒測了
# -----------------------------------------------
# THIS IS A CLASSIC CNN (see examples, section 3)
# -----------------------------------------------
# Note that a few elements have changed (usage of queues).

# Parameters
learning_rate = 0.001
num_steps = 10000
batch_size = 128
display_step = 100

# Network Parameters
dropout = 0.75 # Dropout, probability to keep units

# Build the data input
# X, Y = read_images(DATASET_PATH, MODE, batch_size)


# Create model
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out


# Because Dropout have different behavior at training and prediction time, we
# need to create 2 distinct computation graphs that share the same weights.

# Create a graph for training
logits_train = conv_net(X, N_CLASSES, dropout, reuse=False, is_training=True)
# Create another graph for testing that reuse the same weights
logits_test = conv_net(X, N_CLASSES, dropout, reuse=True, is_training=False)

# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits_train, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Saver object
saver = tf.train.Saver()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Start the data queue
    tf.train.start_queue_runners()

    # Training cycle
    for step in range(1, num_steps+1):

        if step % display_step == 0:
            # Run optimization and calculate batch loss and accuracy
            _, loss, acc = sess.run([train_op, loss_op, accuracy])
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
        else:
            # Only run the optimization op (backprop)
            sess.run(train_op)

    print("Optimization Finished!")

    # Save your model
    saver.save(sess, 'my_tf_model')
