
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from keras import backend as K
from keras.optimizers import Adam
from keras import losses
from keras.utils import np_utils, generic_utils
import numpy as np
import scipy as sp
import random
import scipy.io
from scipy.stats import mode

Experiments = 1

batch_size = 128
nb_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3


print(keras.__version__)

for e in range(Experiments):

	print('Experiment Number ', e)


	# the data, shuffled and split between tran and test sets
	(X_train_All, y_train_All), (X_test, y_test) = mnist.load_data()

	X_train_All = X_train_All.reshape(X_train_All.shape[0], 1, img_rows, img_cols)
	X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
	input_shape = (1, img_rows, img_cols)
	if K.image_data_format() == 'channels_first':
		X_train_All = X_train_All.reshape(X_train_All.shape[0], 1, img_rows, img_cols)
		X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
		input_shape = (1, img_rows, img_cols)
	else:
		X_train_All = X_train_All.reshape(X_train_All.shape[0], img_rows, img_cols, 1)
		X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
		input_shape = (img_rows, img_cols, 1)
	random_split = np.asarray(random.sample(range(0,X_train_All.shape[0]), X_train_All.shape[0]))

	X_train_All = X_train_All[random_split, :, :, :]
	y_train_All = y_train_All[random_split]

	X_valid = X_train_All[10000:15000, :, :, :]
	y_valid = y_train_All[10000:15000]

	X_Pool = X_train_All[20000:60000, :, :, :]
	y_Pool = y_train_All[20000:60000]

	pidx_0 = np.array(np.where(y_Pool==0)).T
	pidx_0 = pidx_0[0:(pidx_0.shape[0]//10),0]
	pX_0 = X_Pool[pidx_0,:,:,:]
	py_0 = y_Pool[pidx_0]
	pidx_1 = np.array(np.where(y_Pool==1)).T
	pidx_1 = pidx_1[0:(pidx_1.shape[0]//10),0]
	pX_1 = X_Pool[pidx_1,:,:,:]
	py_1 = y_Pool[pidx_1]
	pidx_2 = np.array(np.where(y_Pool==2)).T
	pidx_2 = pidx_2[0:(pidx_2.shape[0]//10),0]
	pX_2 = X_Pool[pidx_2,:,:,:]
	py_2 = y_Pool[pidx_2]
	pidx_3 = np.array(np.where(y_Pool==3)).T
	pidx_3 = pidx_3[0:(pidx_3.shape[0]//10),0]
	pX_3 = X_Pool[pidx_3,:,:,:]
	py_3 = y_Pool[pidx_3]
	pidx_4 = np.array(np.where(y_Pool==4)).T
	pidx_4 = pidx_4[0:(pidx_4.shape[0]//10),0]
	pX_4 = X_Pool[pidx_4,:,:,:]
	py_4 = y_Pool[pidx_4]
	pidx_5 = np.array(np.where(y_Pool==5)).T
	pidx_5 = pidx_5[0:(pidx_5.shape[0]//10),0]
	pX_5 = X_Pool[pidx_5,:,:,:]
	py_5 = y_Pool[pidx_5]
	pidx_6 = np.array(np.where(y_Pool==6)).T
	pidx_6 = pidx_6[0:(pidx_6.shape[0]//10),0]
	pX_6 = X_Pool[pidx_6,:,:,:]
	py_6 = y_Pool[pidx_6]
	pidx_7 = np.array(np.where(y_Pool==7)).T
	pidx_7 = pidx_7[0:(pidx_7.shape[0]//10),0]
	pX_7 = X_Pool[pidx_7,:,:,:]
	py_7 = y_Pool[pidx_7]
	pidx_8 = np.array(np.where(y_Pool==8)).T
	pidx_8 = pidx_8[0:(pidx_8.shape[0]//1),0]
	pX_8 = X_Pool[pidx_8,:,:,:]
	py_8 = y_Pool[pidx_8]
	pidx_9 = np.array(np.where(y_Pool==9)).T
	pidx_9 = pidx_9[0:(pidx_9.shape[0]//1),0]
	pX_9 = X_Pool[pidx_9,:,:,:]
	py_9 = y_Pool[pidx_9]

	X_Pool = np.concatenate((pX_0,pX_1,pX_2,pX_3,pX_4,pX_5,pX_6,pX_7,pX_8,pX_9),axis=0)
	y_Pool = np.concatenate((py_0,py_1,py_2,py_3,py_4,py_5,py_6,py_7,py_8,py_9),axis=0)
	random_pool_split = np.asarray(random.sample(range(0,X_Pool.shape[0]),X_Pool.shape[0]))
	X_Pool = X_Pool[random_pool_split,:,:,:]
	y_Pool = y_Pool[random_pool_split]

	X_train = X_Pool
	y_train = y_Pool

	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_valid = X_valid.astype('float32')
	X_Pool = X_Pool.astype('float32')
	X_train /= 255
	X_valid /= 255
	X_Pool /= 255
	X_test /= 255

	Y_test = np_utils.to_categorical(y_test, nb_classes)
	Y_valid = np_utils.to_categorical(y_valid, nb_classes)
	Y_Pool = np_utils.to_categorical(y_Pool, nb_classes)
	Y_train = np_utils.to_categorical(y_train, nb_classes)

	
	c = 3.5
	Weight_Decay = c / float(X_train.shape[0])

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, kernel_regularizer = regularizers.l2(Weight_Decay),activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_classes, activation='softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
	hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_valid, Y_valid))

	print('Evaluating Test Accuracy Without Acquisition')
	score = model.evaluate(X_test, Y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
