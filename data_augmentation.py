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
	
	if tf.contrib.framework.is_tensor(image):
		return image
	else:
		return tf.convert_to_tensor(image)
	
def get_random_augmentation_combinations(length):
	out = [True,False]
	return [random.choice(out) for i in xrange(length)]
	
def augment_data(image_tensor,label_tensor,sess,naug=3):
	label_tensor = tf.reshape(label_tensor,list(label_tensor.get_shape())+[1])
	images_array = np.array(image_tensor.eval(session=sess))
	labels_array = np.array(label_tensor.eval(session=sess))
	#print 'in aug la:',labels_array
	#print labels_array.shape
	labels_array = np.reshape(labels_array,(-1,1))
	#print labels_array.shape
	#print 'Each image::',image_array[0].shape
	nImg = images_array.shape
	#print nImg
	augtime = time.time()
	for na in xrange(naug):
		#print 'Augmentation Step :',na+1
		stime = time.time()
		for nimg in xrange(nImg[0]):
			args = get_random_augmentation_combinations(7)
			transformed_image = get_transformed_image(images_array[nimg],args[0],args[1],args[2],args[3],args[4],args[5],args[6])
			#print transformed_image
			transformed_image = tf.reshape(transformed_image,[1]+list(transformed_image.shape))
			#print transformed_image.shape
			image_tensor = tf.concat([image_tensor,transformed_image],axis=0)
			#print image_tensor.shape
			label_tensor = tf.concat([label_tensor,tf.reshape(tf.convert_to_tensor(labels_array[nimg]),[1,1])],axis=0)
		#print time.time() - stime,'seconds'
		#print image_tensor.get_shape()
	#print tf.contrib.framework.is_tensor(images_tensor)
	print 'Total time taken for augmentation:',time.time()-augtime,' seconds'
	return image_tensor,label_tensor
