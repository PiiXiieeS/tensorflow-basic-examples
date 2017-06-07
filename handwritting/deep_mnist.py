import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# create input object which reads from MNIST dataset. Performs one-hot encoding to define the digit
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

# Using interaction session makes it the default session so we don not need to pass session
sess = tf.InteractiveSession()

#define placeholder
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# change MNIST input data from a list of values to a 28 x 28 x 1 value cube
# which the CNN can use
x_image = tf.reshape(x, [-1, 28, 28, 1], name="x_image")

# define helper function to created weights and biases variables, and convolution, and pooling layers
# we are using RELU as our activation function. These must be initialized to a small positive number
# and with some noise so you don't end up going to zero when comparing diffs
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convulition and pooling = we do convolution, and then pooling to control overfitting
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1], padding='SAME')

# define layers in the NN
#
### 1st convolutional layers
# 32 features for each 5X5 patch of the images
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# do convulition on images, add bias, and push through RELU activation
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
#take results and run through max_pool
h_pool1 = max_pool_2x2(h_conv1)



### 2nd convolutional layers
# process the 32 features from convolutional layer 1, in 5 x 5 patch, return 64 features weights and biases
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

# do convulition on of the output of hte 1st conv layer. Pool results
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


### Fully connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

#connect output of pooling layer 2 as input to fully ocnnected layer
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout some neurons to reduce overfitting
keep_prob = tf.placeholder(tf.float32) # get dropout probability as training input
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


### readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

# define model
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# define loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

# loss optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# what is correct
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# how accurate
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#initialized
sess.run(tf.global_variables_initializer())

### train the model
import time

#define steps and progress display
num_steps = 3000
display_every = 100

#start timer
start_time = time.time()
end_time = time.time()

for i in range(num_steps):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    if i%display_every == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0
        })
        end_time = time.time()
        print("Step {0}, elapsed time {1:.2f} seconds, training accuracy {2:.3f}%".format(i, end_time - start_time, train_accuracy*100))

# display summary
end_time = time.time()

print("Total time for {0} batches: {1:.2f} seconds".format(i+1, start_time))

#accuracy on test data
print("Test accuracy {0:.3f}%".format(accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0
})*100))

sess.close()
