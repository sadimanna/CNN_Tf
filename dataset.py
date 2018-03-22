import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from get_stl import *
from data_augmentation import augment_data
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)

def create_ds_from_tensor(images_tensor,labels_tensor):
	#Creating Dataset from images and labels
	stl_ds = tf.data.Dataset.from_tensor_slices({"images":images_tensor,"labels":labels_tensor})
	return stl_ds

def get_dataset():
	#Getting the Images as Numpy Arrays
	print 'Getting Images as NumPy arrays...'
	image_array = get_all_train_images()
	label_array = get_all_train_labels()
	print 'Image Array shape::',image_array.shape
	print 'Labels Array shape::',label_array.shape
	'''
	#Get Augmented Images
	print 'Data Augmentation in progress...'
	aug_time = time.time()
	images_tensor,labels_tensor = augment_data(image_array,label_array,3)
	print 'Time to get augmented data::',(time.time()-aug_time)/60,'minutes...'
	print 'Augmented Imges shape:',images_tensor.shape,'Labels shape:', labels_tensor.shape
	'''
	#Create Tensor from numpy array
	images_tensor = tf.convert_to_tensor(image_array)
	labels_tensor = tf.convert_to_tensor(label_array)
	
	#Create Dataset from Tensor
	print 'Creating Dataset from Tensors'
	dataset = create_ds_from_tensor(images_tensor,labels_tensor)
	
	return dataset
