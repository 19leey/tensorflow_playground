#MNIST data analysis using deep neural network analysis
#	simple analysis did not account for 'image' analysis
#		location/position matters in images
#	add convolution network layer to handle images - 'moving and filtering (pixel positions)'
#		inspects subsets of image
#		learn features of image (curves)
#		often paired with pool layer
#			generalize each digit shape
#	back propagation - update based on accuracy/results

#imports
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#use TF helper function to import MNIST data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

#interactive session - don't need to pass sess
sess = tf.InteractiveSession()

#define placeholders for MNIST data
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#reshape MNIST data back into 28 x 28 pixel x 1 grayscale value cube to be used by convolution NN
#	'-1' - used to flatten shape or infer shape
#	in our case 'infer shape' -  don't know how many images
x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')

#define helper functions
#RELU activation function
#	if x <= 0, then x = 0
#	if x > 0, then x = x
#truncated_normal - random values from a truncated normal distribution
#	random positive values (in regards to RELU)
#	stddev=0.1 - adds noise so that difference != 0
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(inital)

#constant - some constant (0.1 in this case)
#	0.1 > 0 (in regards to RELU)
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

#convolution and pooling
#	pooling after convolution to help control overfitting
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')




