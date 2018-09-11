#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alexandre Coninx
    ISIR CNRS/UPMC
    06/07/2016
""" 

import numpy as np

from keras.models import Sequential, Model

from keras.layers import Input, merge
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.optimizers import SGD
from keras.callbacks import Callback

from collections import OrderedDict


import gzip, cPickle as pickle

import string

import re

import os

import datetime

from PIL import Image, ImageOps

import cnn_new

import gen_ydata

#matcher = re.compile(".*\/resized-(A?B?B?C?D?)-IMG.*.jpg")
matcher = re.compile(".*\/(\d+_)?([A-Z][A-Z])(-[A-Z][A-Z])?(-[A-Z][A-Z])?(-[A-Z][A-Z])?")


base_data_path = "."
output_path = "./save-big-12-FCN"


vgg16_path = "%s/vgg16_weights.h5" % base_data_path


weightspath = "%s/weights" % output_path
historyfile = "%s/final.gz" % output_path
logfile = "%s/log-big.txt" % output_path

traindump = "%s/train-ds-dump.txt" % output_path
validdump = "%s/valid-ds-dump.txt" % output_path


dsdir = "data_mehdi/DATA"
dslist = "data_mehdi/DATA/ds_12.txt"

vgg_rgboffsets = [103.939, 116.779, 123.68]




#default_trained_weights= "./bestweights-21-07.h5"
#default_trained_weights= "./best-new-interrupted.h5"
default_trained_weights = "./lastweights-fullnet.h5"


def get_now():
	now = datetime.datetime.now()
	tstamp = now.strftime("%d-%m-%y--%H-%M")
	return tstamp
	
	
#classes = OrderedDict([("IMG_1903","oeuf-bio"),
#    ("IMG_1904","sel-baleine"),
#    ("IMG_1905","panzani-farfalle"),
#    ("IMG_1907","tipiak-couscous"),
#    ("IMG_1908","unclebens-riz"),
#    ("IMG_1911","comte-rape"),
##    ("IMG_1912","bonnemaman-confiture"),
#    ("IMG_1913","maille-moutarde"),
#    ("IMG_1914","beurre-tendre"),
#    ("IMG_1915","blanc-poulet-25pct-fleurymichon"),
#    ("IMG_1916","filet-poulet-braise-fleurymichon")])

#classes = ['A','B','C','D']
classes = ['A' + c for c in list(string.ascii_uppercase)] + ['B' + c for c in list(string.ascii_uppercase)[:4]]

#cnames = classes.values()
#cdirs = classes.keys()


nclasses = len(classes)

def decode_classes(g):
	c1 = g[1]
	if(g[2]):
		c2 = g[2][1:]
	else:
		return [c1]
	if(g[3]):
		c3 = g[3][1:]
	else:
		return [c1,c2]
	if(g[4]):
		c4 = g[4][1:]
	else:
		return [c1,c2,c3]
	return [c1,c2,c3,c4]


def classes_to_y(cset):
	y = [0]*nclasses
	for c in cset:
		i = classes.index(c)
		y[i]=1
	return np.array(y,dtype=np.float32)



def gen_ds_list(listfile=dslist):
	dataset = list()
	tags_to_labels, labels_to_tags = gen_ydata.load_mappings()
	with open(listfile,'r') as fd:
		for line in fd:
			[sample_dir_path, s_n_imgs] = line.strip().split('\t')
			n_imgs = int(s_n_imgs)
			m = matcher.match(sample_dir_path)
			if(m):
				g = m.groups()
				matching_classes = decode_classes(g)
				y = classes_to_y(matching_classes)
				fullfpath = "%s/%s" % (dsdir, sample_dir_path)
				csvpath = "%s.csv" % fullfpath
				if(os.path.exists(csvpath)):
					#Fully load CSV data
					csvdata = gen_ydata.parse_csv(csvpath, n_imgs, mapping=tags_to_labels)
				else:
					csvdata = None
				dataset.append((fullfpath,n_imgs,y,matching_classes, csvdata))
			else:
				print("WARNING: dir %s does not match pattern" % fpath)
	return dataset


def smart_split_dataset(orig_dslist, split_factor=0.85, n_single=10, n_combi=3, adjust_with_random=True):
	initial_single = int(split_factor*n_single)
	initial_combi = int(split_factor*n_combi)
	train_dict = dict()
	train_dset = list()
	valid_dset = list()
	np.random.shuffle(orig_dslist)
	print "Generating training and validation dsets iwth a target of %.3f%% train" % (split_factor*100)
	for d in orig_dslist:
		labels = "-".join(sorted(d[3]))
		limit = initial_single if(len(d[3])==1) else initial_combi
		if labels not in train_dict.keys():
			train_dict[labels] = 1
			#print "train_dict[%s] == 0++ -> train" % labels
			train_dset.append(d)
		elif train_dict[labels] < limit:
			train_dset.append(d)
			#print "train_dict[%s] == %d++ -> train" % (labels, train_dict[labels])
			train_dict[labels] += 1
		elif train_dict[labels] == limit:
			#print "train_dict[%s] == %d -> valid" % (labels, train_dict[labels])
			valid_dset.append(d)
		else:
			print "ERROR train_dict[%s] == %d" % (labels, train_dict[labels])
	current_pct = float(len(train_dset))/(len(train_dset) + len(valid_dset))
	print "First deterministic pass complete, rate is %.3f%% train" % (current_pct*100)
	if(adjust_with_random and (current_pct < split_factor)):
		delta = int((split_factor - current_pct)*(len(train_dset) + len(valid_dset)))
		print "Moving %d random sets from valid to train to get the right proportion" % delta
		np.random.shuffle(valid_dset)
		train_dset += valid_dset[:delta]
		valid_dset = valid_dset[delta:]
	np.random.shuffle(valid_dset)
	np.random.shuffle(train_dset)
	new_pct = float(len(train_dset))/(len(train_dset) + len(valid_dset))
	print "Return datasets with rate %.3f%% train" % (new_pct*100)
	return(train_dset,valid_dset)


def load_and_process_img(path,resize=True):
	im = Image.open(path)
	if(resize):
		im = ImageOps.fit(im,(224,224),method=Image.ANTIALIAS,centering=(0.5,0.5))
	iarray = np.array(im,dtype=float)
	for i in range(3):
		iarray[:,:,i] -= vgg_rgboffsets[i]
	return iarray # Color is still on channel 3


def get_ds_total_len(dslist):
	total = 0
	cumsums = []
	for d in dslist:
		cumsums.append(total)
		total += d[1]
	return (total,cumsums)

def find_bin(val,cumsums):
	for (i,binlimit) in enumerate(cumsums):
		if val < binlimit:
			return i-1
	return len(cumsums)-1

def build_random_indices(dslist):
	total, cumsums = get_ds_total_len(dslist)
	global_indices = range(total)
	np.random.shuffle(global_indices)
	index_dir = list()
	index_img = list()
	for i in global_indices:
		dir_i = find_bin(i,cumsums)
		index_dir.append(dir_i)
		img_i = i - cumsums[dir_i]
		index_img.append(img_i)
	return (total, index_dir, index_img)


class DSLoader:
	def __init__(self,dslist,shuff=True,exclusive=False,ds_proportion=1.):
		self.dataset = dslist
		self.nsamples = len(self.dataset)
		#self.nimages = get_ds_total_len(self.dataset)
		#self.indices = range(self.nimages)
		#if(shuff):
		#	np.random.shuffle(self.indices)
		self.current_index = 0
		self.exclusive = exclusive
		self.nimages, self.index_dir, self.index_img = build_random_indices(self.dataset)
		self.n_epoch=int(self.nimages*ds_proportion)
		print("Generator for %d inputs ready (%d really used)" % (self.nimages, self.n_epoch))
	
	def reset_index(self):
		self.current_index=0
	
	def reshuffle(self):
		order = range(self.n_epoch)
		np.random.shuffle(order)
		self.index_dir = [self.index_dir[i] for i in order]
		self.index_img = [self.index_img[i] for i in order]
		#self.nimages, self.index_dir, self.index_img = build_random_indices(self.dataset)
		#np.random.shuffle(self.indices)
	
	def get_sample(self, outer_index):
		i = outer_index % self.nimages
		dirpath, _, _, _, csvdata = self.dataset[self.index_dir[i]]
		xdata = load_and_process_img("%s/image-%04d.jpg" %  (dirpath, self.index_img[i]+1)) # Images are numbered from 1
		ydata = gen_ydata.gen_ydata_from_csvdata(csvdata[self.index_img[i]], classes, exclusive=self.exclusive)
		return (xdata,ydata)
		
	def flow(self,batch_size = 10):
		while(True):
			xsamples = []
			ysamples = []
			thisbatchsize = min(batch_size,self.nimages-self.current_index)
			for i in range(self.current_index,self.current_index+thisbatchsize):
				x,y = self.get_sample(i)
				xsamples.append(np.array([x]))
				#ysamples.append(np.expand_dims(np.array([y]),0))
				ysamples.append(np.array([y]))
			ybatch = np.vstack(ysamples)
			xbatch = np.vstack(xsamples)#.transpose((0,3,1,2)) #7/04: use channels-last order

			self.current_index += thisbatchsize
			if(self.current_index >= self.n_epoch):
				self.reshuffle()
				self.reset_index()
			yield (xbatch, ybatch)

def gen_loaders(valid_pct=0.1,proportion=1.,exclusive=False):
	ds = gen_ds_list()
	tds, vds = smart_split_dataset(ds,split_factor=(1.-valid_pct))
	valid_loader = DSLoader(vds,exclusive=exclusive,ds_proportion=proportion)
	train_loader = DSLoader(tds,exclusive=exclusive,ds_proportion=proportion)
	return (train_loader,valid_loader)
	
	
class LrReducerAndSaver(Callback):
	def __init__(self, patience=0, reduce_rate=0.5, reduce_nb=20, verbose=1,reload_if_bad=False,increase_if_good=0.0,output_logfile=logfile,output_weightsdir=weightspath,always_save=True):
		super(Callback, self).__init__()
		self.patience = patience
		self.wait = 0
		self.best_score = np.inf
		self.reduce_rate = reduce_rate
		self.increase_if_good = increase_if_good
		self.current_reduce_nb = 0
		self.reload_if_bad = reload_if_bad
		self.reduce_nb = reduce_nb
		self.verbose = verbose
		self.now = get_now()
		self.last_weights = None
		self.weights_path = output_weightsdir
		self.logfile = output_logfile
		self.always_save = always_save
		with open(self.logfile,'a') as logfd:
			logfd.write("time\tepoch\ttloss\tvloss\tlast_lr\n")

	def on_epoch_end(self, epoch, logs={}):
		current_score = logs.get('val_loss')
		current_trainloss = logs.get('loss')
		lr = self.model.optimizer.lr.get_value()
		with open(self.logfile,'a') as logfd:
			logfd.write("%s\t%d\t%f\t%f\t%f\n" % (get_now(),epoch,current_trainloss,current_score,lr))
		if(self.always_save or current_score < self.best_score):
			self.model.save_weights("%s/%s-weights-epoch%d.h5" % (self.weights_path, self.now, epoch))
		if current_score < self.best_score:
			self.best_score = current_score
			self.wait = 0
			self.last_weights = "%s/%s-weights-epoch%d.h5" % (self.weights_path, self.now, epoch)
			if self.verbose > 0:
				print('---current best loss: %.3f' % current_score)
			if(self.increase_if_good > 0):
				self.model.optimizer.lr.set_value(np.float32(lr*(1.+self.increase_if_good)))
		else:
			if(self.reload_if_bad and self.last_weights):
				print("Reloading weights")
				self.model.load_weights(self.last_weights)
			if self.wait >= self.patience:
				self.current_reduce_nb += 1
				if self.current_reduce_nb <= self.reduce_nb:
					self.model.optimizer.lr.set_value(np.float32(lr*self.reduce_rate))
					if self.verbose > 0:
						print("Epoch %d: reducing lr, setting to %f" % (epoch, lr*self.reduce_rate))
				else:
					if self.verbose > 0:
						print("Epoch %d: early stopping" % (epoch))
					self.model.stop_training = True
				self.wait = 0
			else:
				self.wait += 1


def apply_to_image(network,imgfile):
	iarray = load_and_process_img(imgfile,resize=True)
	x = np.expand_dims(iarray,0)#.transpose((0,3,1,2))
	y = network.predict(x)
	return y[0]


def load_network(weightsfile,exclusive=False):
	net = cnn_new.make_fcn(nclasses=len(classes),dropout=True,no_fc=False,loadweights=False,exclusive=exclusive)
	net.load_weights(weightsfile)
	return net

def eval_image(imagefile,weights=default_trained_weights,exclusive=True):
	print("Loading network...\n")
	net = load_network(weights,exclusive=exclusive)
	print("Processing image...\n")
	out = apply_to_image(net,imagefile)
	order = np.argsort(-out)
	print out.shape
	print("Result:\n")
	for c in order:
		print "%s : %.3f%%\n" % (classes[c], 100*out[c])
#	print("*** Image identified as %s (%.3f%%) ***\n" % (cnames[order[0]],100*out[order[0]]))
#	print("Closest other matches : %s (%.3f%%), %s (%.3f%%) and %s (%.3f%%)" % (cnames[order[1]],100*out[order[1]],cnames[order[2]],100*out[order[2]],cnames[order[3]],100*out[order[3]]))

def dump_dslist(dslist,fname):
	with open(fname,"w") as f:
		for d in dslist:
			f.write("%s\n" % d[0])


def train_net(load_weights=None,valid_pct=0.1,n_epoch=100,batch_size=1,base_lr=1e-4,ds_reduction_factor=1.,patience=1,reload_if_bad=True,increase_if_good=0.05,initial_weights_path=vgg16_path,output_logfile=logfile, output_weights=weightspath,catcross=True):
	model = cnn_new.make_fcn(nclasses=len(classes),dropout=True,exclusive=catcross)
	if(load_weights):
		model.load_weights(load_weights)
	try:
		os.mkdir(output_path)
		os.mkdir(weightspath)
	except OSError:
		print("WARNING:Output dirs seem to already exist. It is probably a bad thing.")
	sgd = SGD(lr=base_lr, momentum=0.9, nesterov=True)
	model.compile(loss=('categorical_crossentropy' if catcross else 'binary_crossentropy'), optimizer=sgd)
	#if(ds_reduction_factor != 1.):
	#	print("WARNING ds_reduction_factor is somewhat broken, do not use for production stuff")
	train_loader, valid_loader = gen_loaders(valid_pct=valid_pct,proportion=ds_reduction_factor, exclusive=catcross)
	dump_dslist(train_loader.dataset,traindump)
	dump_dslist(valid_loader.dataset,validdump)
	gen_train = train_loader.flow(batch_size)
	gen_val = valid_loader.flow(batch_size)
	cb = LrReducerAndSaver(patience=patience,reload_if_bad=reload_if_bad,increase_if_good=increase_if_good,output_logfile=output_logfile,output_weightsdir=output_weights)
	nsteps = int(np.ceil(train_loader.n_epoch/float(batch_size)))
	valsteps = int(np.ceil(valid_loader.n_epoch/float(batch_size)))
	history = model.fit_generator(gen_train, steps_per_epoch=nsteps, epochs=n_epoch,validation_data=gen_val,validation_steps=valsteps,callbacks=[cb])
	return history, train_loader, valid_loader

