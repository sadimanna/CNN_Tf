import numpy as np
import tensorflow as tf
import time, random
import matplotlib.pyplot as plt
from cnn_model import nn_model,loss,optimizer,accuracy
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)

tf.logging.set_verbosity(tf.logging.INFO)

def get_random_rotation_angle():
	return random.randint(-45,45)

def get_random():
	return random.randint(0,2)/10.0
	
def get_new_size():
	new_size = 96 + random.choice([24,16])
	return [new_size,new_size]
	
def get_random_augmentation_combinations(length):
	out = [True,False]
	return [random.choice(out) for i in xrange(length)]

def get_all_images(img_file):
	images = np.fromfile(img_file,dtype=np.uint8).astype(np.float32)
	images = np.reshape(images,(-1,3,96,96))
	images = np.transpose(images,(0,3,2,1))
	print 'Normalizing Inputs...'
	rmean = np.mean(images[:,:,:,0])
	gmean = np.mean(images[:,:,:,1])
	bmean = np.mean(images[:,:,:,2])
	rstd = np.std(images[:,:,:,0])
	gstd = np.std(images[:,:,:,1])
	bstd = np.std(images[:,:,:,2])	
	images[:,:,:,0] = (images[:,:,:,0] - rmean)#/rstd
	images[:,:,:,1] = (images[:,:,:,1] - gmean)#/gstd
	images[:,:,:,2] = (images[:,:,:,2] - bmean)#/bstd
	print 'R_mean:',rmean,'G_mean:',gmean,'B_mean:',bmean
	print 'R_stddev:',rstd,'G_stddev:',gstd,'B_stddev:',bstd
	return images,rmean,gmean,bmean
	
def get_all_labels(label_file):
	labels = np.fromfile(label_file,dtype=np.uint8)
	#print labels.shape
	return labels

def get_val_images(img_file,rmean,gmean,bmean):
	images = np.fromfile(img_file,dtype=np.uint8).astype(np.float32)
	images = np.reshape(images,(-1,3,96,96))
	images = np.transpose(images,(0,3,2,1))
	print 'Normalizing Validation Images...'
	images[:,:,:,0] = (images[:,:,:,0] - rmean)#/rstd
	images[:,:,:,1] = (images[:,:,:,1] - gmean)#/gstd
	images[:,:,:,2] = (images[:,:,:,2] - bmean)#/bstd
	return images

#Create dataset
#Getting the dataset
print 'Getting the data...'
train_data_path = '/floyd/train_X.bin' #/media/siladittya/fdc481ce-9355-46a9-b381-9001613e3422/siladittya/StudyMaterials/ISI/code/ds/stl10_binary
train_label_path = '/floyd/train_y.bin'

train_img_file = open(train_data_path,'rb')
train_label_file = open(train_label_path,'rb')

train_x,rmean,gmean,bmean = get_all_images(train_img_file)
train_y = get_all_labels(train_label_file)

#Getting Validation Dataset
val_img_path = '/floyd/test_X.bin'
val_label_path = '/floyd/test_y.bin'

val_img_file = open(val_img_path,'rb')
val_label_file = open(val_label_path,'rb')

val_x = get_val_images(val_img_file,rmean,gmean,bmean)
val_y = get_all_labels(val_label_file)
print'Getting Validation set from Test set...'
val_x = val_x[:200]
val_y = val_y[:200]

index = np.arange(train_x.shape[0])
#Set seed placeholder for setting a different seed in each epoch
seedin = tf.placeholder(tf.int64,shape=())
#Keep count
count = 0
#........ This part will used to get training data for each epoch during training
init_count = 0
num_epochs = 100
batch_size = 40
numiter = 125
ne = 0
valacc = []
#Create session
feed_images = tf.placeholder(tf.float32,shape=(None,96,96,3))
feed_labels = tf.placeholder(tf.float32,shape=(None,))

aug_img = tf.placeholder(tf.float32,shape=(96,96,3))

logits = nn_model(feed_images,training = True)

cost = loss(logits,feed_labels)

opt_mom = optimizer(lr=0.01)
opt = opt_mom.minimize(cost)

acc = accuracy(logits,feed_labels)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

img_scale_crop = tf.random_crop(tf.image.resize_images(aug_img,get_new_size()),[96,96,3])

img_rand_flip_lr = tf.image.random_flip_left_right(aug_img)

img_rand_flip_ud = tf.image.random_flip_up_down(aug_img)

builder = tf.saved_model.builder.SavedModelBuilder("/output/cnn_model")

while(ne<num_epochs):
	stime = time.time()
	print 'epoch::',ne+1,'...'
	if ne != 0:
		np.random.shuffle(index)
		train_x = train_x[index]
		train_y = train_y[index]
	for niter in xrange(numiter):
		print 'iter..',niter+1
		offset = niter*batch_size
                x_iter, y_iter = np.array(train_x[offset:offset+batch_size,:,:,:]), np.array(train_y[offset:offset+batch_size])
		y_iter=y_iter-1

		print 'Data Augmenting...'
		augtime = time.time()
		for n in xrange(batch_size):
			args = get_random_augmentation_combinations(3)
			if args[0]:
				x_iter[n] = sess.run(img_scale_crop,feed_dict={aug_img:x_iter[n]})
			if args[1]:
				x_iter[n] = sess.run(img_rand_flip_lr,feed_dict={aug_img:x_iter[n]})
			if args[2]: 
				x_iter[n] = sess.run(img_rand_flip_ud,feed_dict={aug_img:x_iter[n]})
		print 'Time for augmentation:: ',time.time()-augtime,' seconds...'	
		#print 'Labels::',nl.reshape([-1])
		feed_trdict={feed_images:x_iter,feed_labels:y_iter}
		#Train
		sess.run(opt,feed_dict=feed_trdict)

		#Calculate accuracy of Validation set
	if (ne+1)%10==0:
		val_acc = sess.run(acc,feed_dict_acc = {feed_images:val_x,feed_labels:val_y})
		print 'Epoch',ne+1,' Validation accuracy::',val_acc
		valacc.append(val_acc)

		if len(valacc)>=3 and (valacc[-1]-valacc[-2])-(valacc[-2]-valacc[-3]) < 10e-4:
			print 'Change in Learning Rate applied...'
			lr=lr/10
			opt_mom = optimizer(lr)
			opt = opt_mom.minimize(cost)

	cc = sess.run(cost,feed_dict=feed_trdict)
	tr_acc = sess.run(acc,feed_dict = feed_trdict)
	print 'Epoch..',ne+1,
	print 'Training cost::',cc,
	print 'Training accuracy::',tr_acc*100,'%'
	print 'Time reqd.::',(time.time()-stime)/60,'mins...'
	init_count+=1
	ne+=1

builder.add_meta_graph_and_variables(sess, ["EVALUATING"])
builder.save()

#close session
sess.close()
