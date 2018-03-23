import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from dataset import get_dataset
from data_augmentation import augment_data
from cnn_model import nn_model,loss,optimizer,accuracy
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)

tf.logging.set_verbosity(tf.logging.INFO)

def create_next_batch_iterator(dataset,seedin = 6,batch_size = 10):
	#Setting batch size
	batch_size = batch_size
	#Shuffling the dataset
	shuffled_ds = dataset.shuffle(100,seed = seedin,reshuffle_each_iteration = True)
	#Getting batch_size number of images
	batch = shuffled_ds.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
	#Creating a iterator for the dataset
	#iterator = batch.make_one_shot_iterator()
	iterator = batch.make_initializable_iterator()
	#Getting the next batch
	#next_batch = iterator.get_next())
	return iterator

def get_next_batch(iterator):
	next_batch = iterator.get_next()
	return tf.Variable(next_batch['images']),tf.Variable(next_batch['labels'])

#Create session
feed_images = tf.placeholder(tf.float32,shape=(None,96,96,3))
feed_labels = tf.placeholder(tf.float32,shape=(None,1))
sess = tf.Session()
#Creating the CNN Architecture Model
#model = get_model()
#Create dataset
dataset = get_dataset()
#Set seed placeholder for setting a different seed in each epoch
seedin = tf.placeholder(tf.int64,shape=())
#Get iterator
iterator = create_next_batch_iterator(dataset,seedin)
#Initialize the Iterator
#sess.run(iterator.initializer,feed_dict={seedin:6})
'''
#Get next batch
next_batch = get_next_batch(iterator)
next_batch_images = tf.Variable(next_batch['images'])
next_batch_labels = tf.Variable(next_batch['labels'])
sess.run(tf.global_variables_initializer())
'''
'''
print '1',sess.run(next_batch_labels)
print '2',next_batch_labels.eval(session=sess)

print '3',sess.run(next_batch_labels)
print '4',next_batch_labels.eval(session=sess)

print '5',sess.run(next_batch_labels)
print '6',next_batch['labels'].eval(session=sess)
'''
'''
next_batch = augment_data(next_batch_images,next_batch_labels,sess)
print next_batch
print next_batch[1].get_shape().as_list()
'''
#Keep count
count = 0
#........ This part will used to get training data for each epoch during training
init_count = 0
num_epochs = 100
batchsize = 10
numiter = 50
logits = nn_model(feed_images)
cost = loss(logits,feed_labels)
opt_adam = optimizer(cost)
acc = accuracy(logits,feed_labels)
while(ne<num_epochs):
	#Initialize the Iterator
	sess.run(iterator.initializer,feed_dict={seedin:init_count})
	for niter in xrange(numiter):
		#Get next training batch
		next_batch = get_next_batch(iterator)
		next_batch_images = tf.Variable(next_batch['images'])
		next_batch_labels = tf.Variable(next_batch['labels'])
		feed_trdict={images:next_batch_images,labels:next_batch_labels}
		#Train
		sess.run(opt_adam,fed_dict=feed_trdict)
		#Get next Validation batch
		#........................
		#Calculate accuracy of Validation set
		if (niter+1)%10==0:
			val_acc = sess.run(acc,feed_dict_acc = {true_labels:feed_labels}
	init_count+=1
	ne+=1
#close session
sess.close()
