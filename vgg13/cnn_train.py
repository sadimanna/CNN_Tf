import numpy as np
import tensorflow as tf
import time, random
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", FutureWarning)

tf.logging.set_verbosity(tf.logging.INFO)

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

def loss(logits,labels):
	labels = tf.reshape(tf.cast(labels,tf.int64),[-1])
	#print labels.get_shape().as_list(),logits.get_shape().as_list()
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits,name='cross_entropy_per_example')
	cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
	total_loss = tf.add(tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),cross_entropy_mean,name='total_loss')
	return total_loss

def accuracy(logits,true_labels):
	pred_labels = tf.argmax(logits,1)
	true_labels = tf.cast(true_labels,tf.int64)
	#print pred_labels.get_shape().as_list(),true_labels
	correct_pred = tf.cast(tf.equal(pred_labels, true_labels), tf.float32)
	accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
	return accuracy
	
def get_new_size():
	new_size = 96 + random.choice([24,16,0])
	return [new_size,new_size]
	
def get_random_augmentation_combinations(length):
	out = [False,True]
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

def get_test_images(img_file,rmean,gmean,bmean):
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
test_img_path = '/floyd/test_X.bin'
test_label_path = '/floyd/test_y.bin'

test_img_file = open(test_img_path,'rb')
test_label_file = open(test_label_path,'rb')

test_x = get_test_images(test_img_file,rmean,gmean,bmean)
test_y = get_all_labels(test_label_file)
print'Getting Validation set from Test set...'
val_x = test_x[300:500]
val_y = test_y[300:500]
val_y = val_y-1 #Label values converted from [1,10] to [0,10)

index = np.arange(train_x.shape[0])
#Set seed placeholder for setting a different seed in each epoch
seedin = tf.placeholder(tf.int64,shape=())

#Keep count
count = 0
#........ This part will used to get training data for each epoch during training
init_count = 0
num_epochs = 100
batch_size = 50
numiter = 100
ne = 0
valacc = []
#Create session
feed_images = tf.placeholder(tf.float32,shape=(None,96,96,3))
feed_labels = tf.placeholder(tf.float32,shape=(None,))
lr = tf.placeholder(tf.float32,shape=())
keep_prob = tf.placeholder(tf.float32,shape=())

aug_img = tf.placeholder(tf.float32,shape=(96,96,3))
with tf.device('/gpu:0'):
	conv1 = conv2d(feed_images,'conv1',[3,3,3,64],1)
	conv2 = conv2d(conv1,'conv2',[3,3,64,64],1)
	pool1 = maxpool(conv2,'pool1',2,2)
	#size = [N,48,48,64]
	conv3 = conv2d(pool1,'conv3',[3,3,64,128],1)
	conv4 = conv2d(conv3,'conv4',[3,3,128,128],1)
	pool2 = maxpool(conv4,'pool2',2,2)
	#size = [N,24,24,128]
	conv5 = conv2d(pool2,'conv5',[3,3,128,256],1)
	conv6 = conv2d(conv5,'conv6',[3,3,256,256],1)
	pool3 = maxpool(conv6,'pool3',2,2)
	#size = [N,12,12,256]
	conv7 = conv2d(pool3,'conv7',[3,3,256,512],1)
	conv8 = conv2d(conv7,'conv8',[3,3,512,512],1)
	pool4 = maxpool(conv8,'pool4',2,2)
	#size = [N,6,6,512]
	conv9 = conv2d(pool4,'conv9',[3,3,512,512],1)
	conv10 = conv2d(conv9,'conv10',[3,3,512,512],1)
	pool5 = maxpool(conv10,'pool5',2,2)
	#size = [N,3,3,512]
	pool5shape = pool5.get_shape().as_list()
	N = pool5shape[1]*pool5shape[2]*pool5shape[3]
	flattened_pool5 = tf.contrib.layers.flatten(pool5)
	fc1 = tf.contrib.layers.fully_connected(flattened_pool5,2048,weights_regularizer=tf.contrib.layers.l2_regularizer(tf.constant(0.0005, dtype=tf.float32)))
	dropout1 = tf.nn.dropout(fc1,keep_prob)
	fc2 = tf.contrib.layers.fully_connected(dropout1,2048,weights_regularizer=tf.contrib.layers.l2_regularizer(tf.constant(0.0005, dtype=tf.float32)))
	dropout2 = tf.nn.dropout(fc2,keep_prob)
	logits = tf.contrib.layers.fully_connected	(dropout2,10,activation_fn=None,weights_regularizer=tf.contrib.layers.l2_regularizer(tf.constant(0.0005, dtype=tf.float32)))

	cost = loss(logits,feed_labels)

	opt_mom = tf.train.MomentumOptimizer(learning_rate=lr,momentum=0.9)
	opt = opt_mom.minimize(cost)

	acc = accuracy(logits,feed_labels)

img_scale_crop = tf.random_crop(tf.image.resize_images(aug_img,get_new_size()),[96,96,3])
img_rand_flip_lr = tf.image.random_flip_left_right(aug_img)
img_rand_flip_ud = tf.image.random_flip_up_down(aug_img)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

builder = tf.saved_model.builder.SavedModelBuilder("/output/cnn_model_final")

while(ne<num_epochs):
	stime = time.time()
	print 'epoch::',ne+1,'...'
	if ne != 0:
		np.random.shuffle(index)
		train_x = train_x[index]
		train_y = train_y[index]
	for niter in xrange(numiter):
		if (niter+1)%25==0:
			print 'iter..',niter+1
		offset = niter*batch_size
                x_iter, y_iter = np.array(train_x[offset:offset+batch_size,:,:,:]), np.array(train_y[offset:offset+batch_size])
		y_iter=y_iter-1

		#print 'Data Augmenting...'
		#augtime = time.time()
		for n in xrange(batch_size):
			args = get_random_augmentation_combinations(3)
			if args[0]:
				x_iter[n] = sess.run(img_scale_crop,feed_dict={aug_img:x_iter[n]})
			if args[1]:
				x_iter[n] = sess.run(img_rand_flip_lr,feed_dict={aug_img:x_iter[n]})
			if args[2]: 
				x_iter[n] = sess.run(img_rand_flip_ud,feed_dict={aug_img:x_iter[n]})
		#print 'Time for augmentation:: ',time.time()-augtime,' seconds...'
		if ne<40:	
			feed_trdict={feed_images:x_iter,feed_labels:y_iter,lr:0.01,keep_prob:0.5}
		elif ne>=40 and ne<70:
			feed_trdict={feed_images:x_iter,feed_labels:y_iter,lr:0.001,keep_prob:0.5}
		else:
			feed_trdict={feed_images:x_iter,feed_labels:y_iter,lr:0.0001,keep_prob:0.5}
		#Train
		sess.run(opt,feed_dict=feed_trdict)

	#Calculate accuracy of Training set
	cc = sess.run(cost,feed_dict=feed_trdict)
	tr_acc = sess.run(acc,feed_dict = feed_trdict)
	#Calculate accuracy of Validation set
	val_loss = sess.run(cost,feed_dict = {feed_images:val_x,feed_labels:val_y,keep_prob:1.0})
	val_acc = sess.run(acc,feed_dict = {feed_images:val_x,feed_labels:val_y,keep_prob:1.0})
	valacc.append(val_acc)
	
	print 'Epoch..',ne+1,'...'
	print 'Training cost::',cc,
	print 'Training accuracy::',tr_acc*100,'%'
	print 'Validation accuracy::',val_acc*100,'%'
	print 'Time reqd.::',(time.time()-stime)/60,'mins...'
	print '{"metric":"Training Accuracy","value":%f}' % (tr_acc*100)
	print '{"metric":"Training Loss","value":%f}' % cc
	print '{"metric":"Validation Accuracy","value":%f}' % (val_acc*100)
	print '{"metric":"Validation Loss","value":%f}' % val_loss

	init_count+=1
	ne+=1

builder.add_meta_graph_and_variables(sess, ["EVALUATING"])
builder.save()

#PREDICT
test_x = test_x[1:300]
test_y = test_y[1:300]
test_y = test_y-1
index = np.arange(test_x.shape[0])
for nepoch in xrange(100):
	stime = time.time()
	if nepoch !=0:
		np.random.shuffle(index)
		test_x = test_x[index]
		test_y = test_y[index]
	test_loss = sess.run(cost,feed_dict = {feed_images:test_x,feed_labels:test_y,keep_prob:1.0})
	test_acc = sess.run(acc,feed_dict = {feed_images:test_x,feed_labels:test_y,keep_prob:1.0})
	print 'Epoch..',nepoch+1,'...'
	print 'Time reqd.::',(time.time()-stime)/60,'mins...'
	print '{"metric":"Test Accuracy","value":%f}' % (test_acc*100)
	print '{"metric":"Test Loss","value":%f}' % test_loss

#close session
sess.close()
