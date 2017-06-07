import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# using TF helper function to pull down the data from MNIST site
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

# x is a placeholder for the 28 x 28 image data
x = tf.placeholder(tf.float32, shape=[None, 784])

# y_ is called "y bar" and is a 10 element vector, containing the predicted probability of each
# digit (0-9) class. Such as [0.14, 0.8, 0, 0, ...]
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# define weight and blanaces
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# define the model
y = tf.nn.softmax(tf.matmul(x, W) + b)

# loss is cross entropy
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
)

# each training step in gradient descent we want to minimze cross entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# initilize global cvariables
init = tf.global_variables_initializer()

# create an interactive session that can span multiple code blocks. Don't
# forget to explicity close the session with session.close()
session = tf.Session()

# perform the initilization wich is only the intiialization of all global variables
session.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# evaluate how well the model did. Do this by comparing the digit with the highest probability
# actual(y) and predicted (y_)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
test_accuracy = session.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
print("Test accuracy: {0}%".format(test_accuracy*100.0))
session.close()
