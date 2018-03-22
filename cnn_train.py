import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from dataset import get_dataset
from data_augmentation import augment_data
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)


def create_next_batch_iterator(dataset,seedin = 6,batch_size = 4):
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
sess.run(iterator.initializer,feed_dict={seedin:6})
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

while True:
	try:
		print(sess.run(next_batch))
		count+=1
	except tf.errors.OutOfRangeError:
		print 'End of Dataset'
		#break
		init_count+=1
		sess.run(iterator.initializer,feed_dict={seedin:init_count})
		print init_count,' initialization'
		if init_count==10:
			break
print count



#close session
sess.close()


