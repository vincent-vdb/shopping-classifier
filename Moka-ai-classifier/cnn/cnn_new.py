#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alexandre Coninx
    ISIR CNRS/UPMC
    07/04/2017
""" 

import numpy as np
import vgg16_base

from keras.models import Model
from keras.layers.merge import Concatenate
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.layers.core import Flatten, Dense, Dropout

from keras.layers.convolutional import Deconvolution2D # aka Conv2DTranspose


def make_cnn(nclasses=30,dropout=False,loadweights=True):
	#Get a pretrained topless VGG16
	w = "imagenet" if loadweights else None
	vgg = vgg16_base.VGG16(include_top=False,input_shape=(224, 224, 3),weights=w)
	# Classification block
	x = Flatten(name='flatten')(vgg.get_layer("block5_pool").output)
	x = Dense(4096, activation='relu', name='fc1')(x)
	if(dropout):
		x = Dropout(0.5)(x)
	x = Dense(4096, activation='relu', name='fc2')(x)
	if(dropout):
		x = Dropout(0.5)(x)
	final = Dense(nclasses, activation='sigmoid', name='predictions')(x)
	# Make model
	cnn_model = Model(inputs=vgg.input, outputs=final)
	return cnn_model

#Work with about 6GB mem
n_fc_nodes_1 = 1024#4096#1024
n_fc_nodes_2 = 1024#4096#512
n_fc_nodes_inter = 128 #4096#256
#n_fc_nodes_inter = None #256 #4096#256




def make_fcn(nclasses=30,dropout=False,loadweights=True,no_fc=False,exclusive=False):
	#Get a pretrained topless VGG16
	w = "imagenet" if loadweights else None
	vgg = vgg16_base.VGG16(include_top=False,input_shape=(224, 224, 3),weights=w)
	deconvs = []
	for i in range(1,6): # 3 top layers only
		if(n_fc_nodes_inter):
			x =  Convolution2D(n_fc_nodes_inter, (1, 1), activation='relu', name="stage%d_deconv" % i)(vgg.get_layer("block%d_pool" % i).output)
		else:
			x = vgg.get_layer("block%d_pool" % i).output
		deconvs.append(UpSampling2D(size=(2**i, 2**i), name="stage%d_upsample" % i)(x))
	# Merge deconvs
	merged = Concatenate(axis=-1,name="deconvs_concat")(deconvs)
	if(no_fc):
		x=merged
	else:
		# Final steps
		x = Convolution2D(n_fc_nodes_1, (1, 1), activation='relu', name="fc_eq_1")(merged)
		if(dropout):
			x = Dropout(0.5)(x)
		x = Convolution2D(n_fc_nodes_2, (1, 1), activation='relu', name="fc_eq_2")(x)
		if(dropout):
			x = Dropout(0.5)(x)
	nfinal = nclasses if not exclusive else nclasses+1
	final = Convolution2D(nfinal, (1, 1), activation=('sigmoid' if not exclusive else 'softmax'), name="fcn_out")(x)
	# Make model
	fcn_model = Model(inputs=vgg.input, outputs=final)
	return fcn_model



