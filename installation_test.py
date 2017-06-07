import tensorflow as tf

sess = tf.Session()

# verify printing a string
hello = tf.constant("Hello from Tensorflow")
print(sess.run(hello))

# perform some simple math
a = tf.constant(20)
b = tf.constant(22)
print('a + b = {0}'.format(sess.run(a+b)))
