#MNIST data analysis using simple neural network

#imports
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#use TF helper function to import MNIST data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

#x - placeholder for the 28x28 image
#	None - not sure how many images will be inputted
#	784 - each image is 28x28 flattened array or 1x784
x = tf.placeholder(tf.float32, shape=[None, 784])

#y_ - predicted probability of each digit(0-9) class
#	None - not sure how many images
#	10 - only 10 possibilites (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#define weights and bias
#	weight - each pixel has a possibility of 10 different probabilities
W = tf.Variable(tf.zeros([784, 10]))
#	bias - only 10 possibilites for bias (digits)
b = tf.Variable(tf.zeros([10]))

#define model
#	matmul - matrix multipy - order determines shape
#		needed to be x since x has an unknown dimension
#	softmax - activation function
#		softmax often used when determining class of output
#		similar to logistic regression - multiclass
y = tf.nn.softmax(tf.matmul(x, W) + b)

#loss is defined by cross entropy
#	softmax_cross_entropy_with_logits - difference between predictions and actual data
#	reduce_mean - returns 'mean' of the differences
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#define training step to minimize loss(cross entropy)
#	0.5 - learning rate (smaller the better)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#initialize all variables
init = tf.global_variables_initializer()

#create an interactive session
sess = tf.Session()

#run initialization
sess.run(init)

#run through 1000 training steps
for i in range(1000):
#	next_batch - gets 100 random data points from data
#	batch_xs - image
#	batch_ys - digit class
	batch_xs, batch_ys = mnist.train.next_batch(100)

#	optimize the data
#	trace the variables back (placeholder)
#		x - feeds the softmax model function
#		y_ - feeds the cross entropy function
	sess.run(train_step, {x: batch_xs, y_: batch_ys})

#evaluate correctness(accuracy) of model
#compare digit class with highest probability in y and y_
#	probability should be the same - exactly correct
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#accuracy calcuation 
#convert correct prediction(true/false) array to 1s and 0s
#	compute mean of this array - percentage
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#input test data to test out accuracy of trained model
test_accuracy = sess.run(accuracy, {x: mnist.test.images, y_: mnist.test.labels})
print("Accuracy: {0}%".format(test_accuracy * 100))




#Tensor Shapes
#	scalar = 69				[]
#	vector = [1, 2, 3]			[3]
#	matrix = [[1, 2], [3, 4], [5, 6]]		[3, 2]
#		first - number of 'vectors'
#		second - size of each 'vector'
#	cube = ...
#		first - number of matricies
#		second - number of vectors in each matrix
#		third - size of each vector