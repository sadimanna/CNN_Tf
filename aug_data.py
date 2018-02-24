import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as keras
import time, random

def get_random_rotation_angle():
	return random.randint(30,180)

def get_random_wh_shift():
	return random.random()-0.5
	
def get_random_shear():
	return np.pi*random.randint(0,45)/180
	
def get_random_zoom():
	return random.random()

def get_random_flip():
	return random.choice([True,False])

def get_random_function():
	return random.choice([lambda x : tf.image.random_brightness(x,abs(random.random()-0.5)),lambda x : tf.image.random_contrast(x,0.0,1.0)])

def get_image_data_generator():
	return keras.preprocessing.image.ImageDataGenerator(
    rotation_range=get_random_rotation_angle(),\
    width_shift_range=get_random_wh_shift(),\
    height_shift_range=get_random_wh_shift(),\
    shear_range=get_random_shear(),\
    zoom_range=get_random_zoom(),\
    horizontal_flip=get_random_flip(),\
    vertical_flip=get_random_flip(),\
    preprocessing_function=get_random_function(),\
    data_format="channels_last")
    
def augment_data(image_array,label_array):
	print image_array.ndim
	print image_array[0].reshape(1,96,96,3).shape
	images_array = image_array.copy()
	labels_array = label_array.copy()
	#Create a list of various datagenerators with different arguments
	datagenerators = []
	ndg = 10
	for ndata in xrange(ndg):
		datagenerators.append(get_image_data_generator())
	bsize = image_array.shape[0]
	print bsize
	for dgen in datagenerators:
		#dgen.fit(image_array)
		(aug_img,aug_label) = dgen.flow(image_array[0].reshape(1,96,96,3),label_array[0].reshape(1,1),batch_size=bsize,shuffle=True)
		print aug_img.shape
		#images_array = np.concatenate([images_array,aug_img],axis=0)
		#labels_array = np.concatenate([labels_array,aug_label],axis=0)
	return (images_array,labels_array)
	
