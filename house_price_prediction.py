import tensorflow as tf
import numpy as np
import math
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# generating some houses size between 1000 and 3500 (typical sq ft of a house)
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

# generating house prices from house size with a random noise added
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)

# # plot data
# matplotlib.pyplot.plot(house_size, house_price, 'bx')
# matplotlib.pyplot.ylabel("Price")
# matplotlib.pyplot.xlabel("Size")
# matplotlib.pyplot.show()


# normalizing to prevent under/overflows
def normalize(array):
    return (array-array.mean()) / array.std()

# number of training samples (70%).
num_train_samples = int(math.floor(num_house * 0.7))


# define train data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asanyarray(house_price[:num_train_samples:])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

# define test data
test_house_size = np.asarray(house_size[num_train_samples:])
test_house_price = np.asanyarray(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)

# set up the Tensorflow placeholders that get updates as we descent down the gradient
tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float", name="price")

# define the variables holding the size_factor and price we set during training
# we initialize them to some random values  based on the normal distribution

tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

# 2. define the operation for the predicting values - predicted price = (size_factor * house_size) + price_offset
# Notice, the use of the tensorflow add and multiply functions. These add the operations to the computation graph.
# AND the tensorflow methods understand how to deal wwith Tensors. Therefore do not try to use numpy or other library
# methods.
tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)


# 3. define the loss function (how much errro) - mean squared error
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred - tf_price, 2))/(2*num_train_samples)

# optimizer learning rate. the size of steps down the gradient
learning_rate = 0.1

# 4. define a gradient descent optimizer that will minimize the loss defined in the operation "Cost"
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

# initializing variablees
init = tf.global_variables_initializer()

# launch the graph in the session
with tf.Session() as sess:
    sess.run(init)

    # set how often display train progress and number of iterations
    display_every = 2
    num_training_iter = 50

    # calculate number of lines to animation
    fit_num_plots = int(math.floor(num_training_iter/display_every))
    # add storage of factor and offset values from each epoch
    fit_size_factor = np.zeros(fit_num_plots)
    fit_price_offsets = np.zeros(fit_num_plots)
    fit_plot_idx = 0

    # keep iterating the training data
    for iteration in range(num_training_iter):
        # fit all training data
        for (x, y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})

        #display status
        if (iteration+1) % display_every == 0:
            c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
            print("iteration #:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(c), \
                "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))
            # save the fit size_factor nad price_offset to allow animation of learning process
            fit_size_factor[fit_plot_idx] = sess.run(tf_size_factor)
            fit_price_offsets[fit_plot_idx] = sess.run(tf_price_offset)
            fit_plot_idx = fit_plot_idx + 1


    print("Optimization finished")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
    print("Trained cost=", training_cost, "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset), '\n')

    # Plot of training and test data, and learned regression

    # get values used to normalized data so we can denormalize data back to its original scale
    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_price_mean = train_price.mean()
    train_price_std = train_price.std()

    # Plot the graph
    matplotlib.pyplot.rcParams["figure.figsize"] = (10,8)
    matplotlib.pyplot.figure()
    matplotlib.pyplot.ylabel("Price")
    matplotlib.pyplot.xlabel("Size (sq.ft)")
    matplotlib.pyplot.plot(train_house_size, train_price, 'go', label='Training data')
    matplotlib.pyplot.plot(test_house_size, test_house_price, 'mo', label='Testing data')
    matplotlib.pyplot.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
             (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset)) * train_price_std + train_price_mean,
             label='Learned Regression')

    matplotlib.pyplot.legend(loc='upper left')
    matplotlib.pyplot.show()

     #
    # Plot another graph that animation of how Gradient Descent sequentually adjusted size_factor and price_offset to
    # find the values that returned the "best" fit line.
    fig, ax = matplotlib.pyplot.subplots()
    line, = ax.plot(house_size, house_price)

    matplotlib.pyplot.rcParams["figure.figsize"] = (10,8)
    matplotlib.pyplot.title("Gradient Descent Fitting Regression Line")
    matplotlib.pyplot.ylabel("Price")
    matplotlib.pyplot.xlabel("Size (sq.ft)")
    matplotlib.pyplot.plot(train_house_size, train_price, 'go', label='Training data')
    matplotlib.pyplot.plot(test_house_size, test_house_price, 'mo', label='Testing data')

    def animate(i):
        line.set_xdata(train_house_size_norm * train_house_size_std + train_house_size_mean)  # update the data
        line.set_ydata((fit_size_factor[i] * train_house_size_norm + fit_price_offsets[i]) * train_price_std + train_price_mean)  # update the data
        return line,

     # Init only required for blitting to give a clean slate.
    def initAnim():
        line.set_ydata(np.zeros(shape=house_price.shape[0])) # set y's to 0
        return line,

    ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, fit_plot_idx), init_func=initAnim,
                                 interval=1000, blit=True)

    matplotlib.pyplot.show()
