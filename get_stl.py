import numpy as np
import tensorflow as tf
img_height = 96
img_width = 96
img_depth = 3

img_size = img_height*img_width*img_depth

train_data_path = '/media/siladittya/d801fb13-809a-41b4-8f2e-d617ba103aba/ISI/Datasets/stl10_binary/train_X.bin'
train_label_path = '/media/siladittya/d801fb13-809a-41b4-8f2e-d617ba103aba/ISI/Datasets/stl10_binary/train_y.bin'

train_img_file = open(train_data_path,'rb')
train_label_file = open(train_label_path,'rb')

'''
def get_train_img():	
	image = np.fromfile(train_img_file,dtype=np.uint8,count=img_size)
	image = np.reshape(image,(3,96,96))
	image = np.transpose(image,(2,1,0))
	return image

def get_train_labels():
	label = np.fromfile(train_label_file,dtype=np.uint8,count=1)
	return label
'''

def get_all_train_images():
	images = np.fromfile(train_img_file,dtype=np.uint8)
	images = np.reshape(images,(-1,3,96,96))
	images = np.transpose(images,(0,3,2,1))
	print images.shape
	return images
	
def get_all_train_labels():
	labels = np.fromfile(train_label_file,dtype=np.uint8)
	#print labels.shape
	return labels

def get_images_tensor(images_array):
	images_tensor = tf.convert_to_tensor(images_array)
	#sess = tf.Session()
	#print tensor.get_shape()
	#sess.close()
	return images_tensor
	
def get_labels_tensor(labels_array):
	labels_tensor = tf.convert_to_tensor(labels_array)
	return labels_tensor

