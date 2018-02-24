import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as keras
import time, random

def get_random_rotation_angle():
	return random.randint(30,180)

def get_random():
	return random.randint(0,4)/10.0
	
def get_random_shear():
	return random.randint(2,13)/10.0

def get_transformed_image(image,random_rotate=False,random_hor_flip=False,\
random_ver_flip=False,random_wh_shift=False,random_shear=False,random_br=False,random_con=False):
	if random_rotate:
		image = keras.preprocessing.image.random_rotation(image,get_random_rotation_angle(),row_axis=0,col_axis=1,channel_axis=2)
	if random_wh_shift:
		image = keras.preprocessing.image.random_shift(image,get_random(),get_random(),row_axis=0,col_axis=1,channel_axis=2)
	if random_shear:
		image = keras.preprocessing.image.random_shear(image,get_random_shear(),0,1,2)
	if random_hor_flip:
		image = tf.image.random_flip_left_right(image)
	if random_ver_flip:
		image = tf.image.random_flip_up_down(image)
	#if random_zoom:
	#	image = keras.preprocessing.image.random_zoom(image,(get_random_zoom(),get_random_zoom()),row_axis=0,col_axis=1,channel_axis=2)
	if random_br:
		image = tf.image.random_brightness(image,get_random())
	if random_con:
		image = tf.image.random_contrast(image,0.1,0.5)
		
	return image
	
def get_random_augmentation_combinations(length):
	out = [True,False]
	return [random.choice(out) for i in xrange(length)]
	
def augment_data(image_array,label_array,naug=5):
	images_tensor = tf.convert_to_tensor(image_array)
	labels_tensor = tf.convert_to_tensor(label_array)
	#print labels_tensor.shape
	labels_tensor = tf.reshape(labels_tensor,list(labels_tensor.shape)+[1])
	#print labels_tensor.shape
	#print 'Each image::',image_array[0].shape
	for na in xrange(naug):
		print 'Augmentation Step :',na+1
		#stime = time.time()
		for nimg in xrange(image_array.shape[0]):
			args = get_random_augmentation_combinations(7)
			transformed_image = get_transformed_image(image_array[nimg],args[0],args[1],args[2],args[3],args[4],args[5],args[6])
			transformed_image = tf.reshape(transformed_image,[1]+list(transformed_image.shape))
			#print transformed_image.shape
			images_tensor = tf.concat([images_tensor,transformed_image],axis=0)
			#print images_tensor.shape
			#print label_array[nimg]
			labels_tensor = tf.concat([labels_tensor,tf.reshape(tf.convert_to_tensor(np.array([label_array[nimg]])),[1,1])],axis=0)
		#print time.time() - stime
	return images_tensor,labels_tensor
