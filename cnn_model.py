import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt


def get_weights(name,shape):
	return tf.get_variable(name=name,shape=shape,initializer = tf.contrib.layers.xavier_initializer(uniform=False),regularizer = tf.contrib.layers.l2_regularizer(tf.constant(0.0005, dtype=tf.float32)))

def get_bias(name,shape):
	return tf.get_variable(name=name,shape=shape,initializer = tf.contrib.layers.xavier_initializer(uniform=False))

def conv2d(inp,name,kshape,s):
	with tf.variable_scope(name) as scope:
		kernel = get_weights('weights',shape=kshape)
		conv = tf.nn.conv2d(inp,kernel,[1,s,s,1],'SAME')
		bias = get_bias('biases',shape=kshape[3])
		preact = tf.nn.add_bias(conv,bias)
		convlayer = tf.nn.relu(preact,name=scope.name)
	return convlayer

def maxpool(inp,name,k,s):
	return tf.nn.max_pool(inp,ksize=[1,k,k,1],strides=[1,s,s,1],padding='SAME',name=name)
	
def dropout(inp,keep_prob,training):
	if training:
		return tf.nn.dropout(inp,keep_prob)
	else:
		return tf.nn.dropout(inp,tf.constant(1.0,dtype=tf.float32))

def fc(inp,name,ksize,N=1):
	with tf.variable_scope(name) as scope:
		if N!=1:
			flattened = tf.reshape(inp,[N,-1])
		else:
			flattened = inp
		dim = flattened.get_shape().as_list()[1]
		weights = get_weights('weights',shape=[dim,ksize])
		biases = get_bias('bias',[ksize])
		fc = tf.nn.relu(tf.add(tf.matmul(flattened,weights),biases),name=scope.name)
	return fc
		
#VGG Net Model
def nn_model(images,labels):
	imgshape = images.get_shape().as_list()
	imgh = imgshape[1]
	imgw = imgshape[2]
	#size = [N,96,96,3]
	images = tf.cast(images,tf.float16)
	labels = tf.cast(labels,tf.float16)
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
	#conv8
	conv8 = conv2d(conv7,'conv8',[3,3,256,256],1)
	#pool3
	pool3 = maxpool(conv8,'pool3',2,2)
	#size = [N,12,12,256]
	#conv9
	conv9 = conv2d(pool3,'conv9',[3,3,256,512],1)
	#conv6
	conv10 = conv2d(conv9,'conv10',[3,3,512,512],1)
	#conv7
	conv11 = conv2d(conv10,'conv11',[3,3,512,512],1)
	#conv8
	conv12 = conv2d(conv11,'conv12',[3,3,512,512],1)
	#pool4
	pool4 = maxpool(conv12,'pool4',2,2)
	#size = [N,6,6,512]
	#conv13
	conv13 = conv2d(pool4,'conv13',[3,3,512,512],1)
	#conv6
	conv14 = conv2d(conv13,'conv14',[3,3,512,512],1)
	#conv7
	conv15 = conv2d(conv14,'conv15',[3,3,512,512],1)
	#conv8
	conv16 = conv2d(conv15,'conv16',[3,3,512,512],1)
	#pool5
	pool5 = maxpool(conv16,'pool5',2,2)
	#size = [N,3,3,512]
	N = imgshape[0]
	#keepprob
	keep_prob = tf.constant(0.5,dtype=tf.float32)
	#fc1
	fc1 = fc(pool5,'fc1',2048,N)
	#dropout1
	dropout1 = dropout(fc1,keep_prob,training=True)
	#fc2
	fc2 = fc(dropout1,'fc2',2048,1)
	#dropout2
	dropout2 = dropout(fc2,keep_prob,training=True)
	#fc3
	fc3 = fc(dropout2,'fc3',11,1)
	#11 since 10 for the trainable classes and 1 extra for the unknown
	return fc3
	
	
