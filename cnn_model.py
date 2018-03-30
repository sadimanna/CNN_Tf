import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt


def get_weights(name,shape):
	return tf.get_variable(name=name,shape=shape,initializer = tf.contrib.layers.xavier_initializer(uniform=False),regularizer = tf.contrib.layers.l2_regularizer(tf.constant(0.0005, dtype=tf.float32)))

def get_bias(name,shape):
	return tf.get_variable(name=name,shape=shape,initializer = tf.zeros_initializer())

def conv2d(inp,name,kshape,s):
	with tf.variable_scope(name) as scope:
		kernel = get_weights('weights',shape=kshape)
		conv = tf.nn.conv2d(inp,kernel,[1,s,s,1],'SAME')
		bias = get_bias('biases',shape=kshape[3])
		preact = tf.nn.bias_add(conv,bias)
		convlayer = tf.nn.relu(preact,name=scope.name)
	return convlayer

def maxpool(inp,name,k,s):
	return tf.nn.max_pool(inp,ksize=[1,k,k,1],strides=[1,s,s,1],padding='SAME',name=name)
	
def dropout(inp,keep_prob,training=False):
	if training:
		return tf.nn.dropout(inp,keep_prob)
	else:
		return tf.nn.dropout(inp,tf.constant(1.0,dtype=tf.float32))
'''
def fc(inp,name,ksize,apply_relu=False):
	with tf.variable_scope(name) as scope:
		dim = inp.get_shape().as_list()[1]
		weights = get_weights('weights',shape=[dim,ksize])
		biases = get_bias('bias',[ksize])
		if apply_relu:
			fc = tf.nn.relu(tf.nn.bias_add(tf.matmul(inp,weights),biases),name=scope.name)
		else:
			fc = tf.nn.bias_add(tf.matmul(inp,weights),biases,name=scope.name)
	return fc
'''		
#VGG16 Net Model
def nn_model(images,training):
	imgshape = images.get_shape().as_list()
	#print imgshape
	imgh = imgshape[1]
	imgw = imgshape[2]
	#size = [N,96,96,3]
	images = tf.cast(images,tf.float32)
	#conv1
	conv1 = conv2d(images,'conv1',[3,3,3,64],1)
	#conv2
	conv2 = conv2d(conv1,'conv2',[3,3,64,64],1)
	#pool1
	pool1 = maxpool(conv2,'pool1',2,2)
	#size = [N,48,48,64]
	#conv3
	conv3 = conv2d(pool1,'conv3',[3,3,64,128],1)
	#conv4
	conv4 = conv2d(conv3,'conv4',[3,3,128,128],1)
	#pool2
	pool2 = maxpool(conv4,'pool2',2,2)
	#size = [N,24,24,128]
	#conv5
	conv5 = conv2d(pool2,'conv5',[3,3,128,256],1)
	#conv6
	conv6 = conv2d(conv5,'conv6',[3,3,256,256],1)
	#conv7
	conv7 = conv2d(conv6,'conv7',[3,3,256,256],1)
	#pool3
	pool3 = maxpool(conv7,'pool3',2,2)
	#size = [N,12,12,256]
	#conv9
	conv8 = conv2d(pool3,'conv8',[3,3,256,512],1)
	#conv6
	conv9 = conv2d(conv8,'conv9',[3,3,512,512],1)
	#conv7
	conv10 = conv2d(conv9,'conv10',[3,3,512,512],1)
	#pool4
	pool4 = maxpool(conv10,'pool4',2,2)
	#size = [N,6,6,512]
	#conv13
	conv11 = conv2d(pool4,'conv11',[3,3,512,512],1)
	#conv6
	conv12 = conv2d(conv11,'conv12',[3,3,512,512],1)
	#conv7
	conv13 = conv2d(conv12,'conv13',[3,3,512,512],1)
	#pool5
	pool5 = maxpool(conv13,'pool5',2,2)
	#size = [N,3,3,512]
	pool5shape = pool5.get_shape().as_list()
	N = pool5shape[1]*pool5shape[2]*pool5shape[3]
	#keepprob
	keep_prob = tf.constant(0.5,dtype=tf.float32)
	#flattened_pool5
	flattened_pool5 = tf.contrib.layers.flatten(pool5)
	#fc1
	fc1 = tf.contrib.layers.fully_connected(flattened_pool5,4096)
	#dropout1
	dropout1 = dropout(fc1,keep_prob,training)
	#fc2
	fc2 = tf.contrib.layers.fully_connected(dropout1,4096)
	#dropout2
	dropout2 = dropout(fc2,keep_prob,training)
	#fc3
	fc3 = tf.contrib.layers.fully_connected(dropout2,10,activation_fn=None)
	#11 since 10 for the trainable classes and 1 extra for the unknown
	return fc3

def loss(logits,labels):
	labels = tf.reshape(tf.cast(labels,tf.int64),[-1])
	#print labels.get_shape().as_list(),logits.get_shape().as_list()
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
	total_loss = tf.add(tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),cross_entropy_mean,name='total_loss')
	return total_loss

def optimizer(lr):
	#return tf.train.AdamOptimizer(learning_rate=lr)
	#return tf.train.GradientDescentOptimizer(learning_rate=lr)
	return tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.9)
	
def accuracy(logits,true_labels):
	pred_labels = tf.argmax(logits,1)
	true_labels = tf.cast(true_labels,tf.int64)
	#print pred_labels.get_shape().as_list(),true_labels
	correct_pred = tf.cast(tf.equal(pred_labels, true_labels), tf.float32)
	accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
	return accuracy
	
