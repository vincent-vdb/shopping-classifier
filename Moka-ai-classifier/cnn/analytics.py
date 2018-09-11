#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alexandre Coninx
    ISIR CNRS/UPMC
    04/06/2017
""" 

import numpy as np

import csv

from classif_mehdi_fcn import classes, classes_to_y

import validate, gen_ydata

import matplotlib.pyplot as plt

from random import randint #

plt.ion()

tags_to_labels, labels_to_tags = gen_ydata.load_mappings()


def parse_videologfile(fname,histo_nbins=10):
	out = list()
	with open(fname,'r') as fd:
		reader = csv.reader(fd,delimiter=',')
		for line in reader:
			i = 0
			n = int(line[i])
			i+=1
			histograms=list()
			for c in classes:
				h = np.array([int(v) for v in line[i:i+histo_nbins]])
				histograms.append(h)
				i += histo_nbins
			classesok = line[i].split('-')
			scores = [float(v) for v in line[i+1:]]
			out.append((n,histograms,classesok,scores))
	return out
			
def parse_bad_simple(fname):
	out = dict()
	with open(fname,'r') as fd:
		fd.readline() #Skip header
		reader = csv.reader(fd,delimiter=',')
		for line in reader:
			imgsplit = line[0].split('/')
			video = imgsplit[-2]
			iframe = int(imgsplit[-1][6:10])
			if video in out.keys():
				out[video].append(iframe)
			else:
				out[video] = [iframe]
	return out


mask_linear = np.arange(10)/10.

mask_square = np.arange(10)**2/float(np.max(np.arange(10)**2))

mask_yes = mask_square
mask_no = mask_yes[::-1]

mask_notsure_nnorm = mask_yes*mask_no
mask_notsure = mask_notsure_nnorm/mask_notsure_nnorm.sum()

def metric_prct_th_mask(h,th=0.01,mask=np.arange(10)/10.):
	return np.sum(mask*h) > th*np.sum(h)


def reeval_vdata_metric(v,trueclasses,metric):
	out = list()
	for frame in v:
		result=list()
		for (i,c) in enumerate(classes):
			histos = frame[1][i]
			if(metric(histos)):
				result.append(c)
		precision,recall,f1 = validate.compute_stats(result,trueclasses)
		out.append((result,[precision,recall,f1]))
	return out


def kalman_filter_basic_argmax(v,alpha=0.5):
	outobs_yes = list()
	outobs_no = list()
	outobs_notsure = list()
	outfilter = list()
	vbuffer = np.zeros(len(classes))
	for frame in v:
		result_y=list()
		result_n=list()
		result_ns=list()
		for (i,c) in enumerate(classes):
			histos = frame[1][i]
			result_y.append(np.sum(mask_yes*histos)/np.sum(histos))
			result_n.append(np.sum(mask_no*histos)/np.sum(histos))
			result_ns.append(np.sum(mask_notsure*histos)/np.sum(histos))
		outobs_yes.append(np.array(result_y))
		outobs_no.append(np.array(result_n))
		outobs_notsure.append(np.array(result_ns))
		vbuffer = vbuffer*(1-alpha) + np.array(result_y)*alpha
		outfilter.append(vbuffer)
	return (outobs_yes,outobs_no,outobs_notsure,outfilter)

def evidence_accumulation(v,keep=1.,add=1.,loss=-0.02):
	outobs = list()
	outfilter = list()
	vbuffer = np.zeros(len(classes))
	for frame in v:
		result=list()
		for (i,c) in enumerate(classes):
			histos = frame[1][i]
			result.append(np.sum(mask_yes*histos)/np.sum(histos))
		outobs.append(np.array(result))
		vbuffer = np.maximum(vbuffer*keep + np.array(result)*add + loss,np.zeros(len(classes)))
		outfilter.append(vbuffer)
	return (outobs,outfilter)



class EAfilter:
	def __init__(self,keep=1.,add=1.,loss=-0.02,plot=False,plotsize=50):
		self.buffer = np.zeros(len(classes))
		self.k = keep
		self.a = add
		self.l = loss
		self.i = 0
		self.plot = plot
		if(plot):
			self.plotsize = plotsize
		self.reset()
		
	def add_data(self,contributions):
		self.buffer = np.maximum(self.buffer*self.k + np.array(contributions)*self.a + self.l,np.zeros(len(classes)))
		if(self.plot):
			self.update_plot()
		self.i += 1
		
	def reset(self):
		self.i = 0
		self.buffer = np.zeros(len(classes))
		if(self.plot):
			self.xvals = np.arange(0)
			self.yvals = np.zeros((0,len(classes)))
			plt.clf()
			self.lines = plt.plot(self.xvals,self.yvals)
			for (i,l) in enumerate(self.lines):
				l.set_label(labels_to_tags[classes[i]])
			plt.show()

	def update_plot(self):
		if(self.i >= self.plotsize):	
			self.xvals = np.arange(self.plotsize) + self.i + 1 - self.plotsize
			self.yvals = np.vstack([self.yvals,self.buffer.reshape((1,len(classes)))])
			self.yvals = self.yvals[1:,:]
		else:
			self.yvals = np.vstack([self.yvals,self.buffer.reshape((1,len(classes)))])
			self.xvals = np.arange(self.i+1)
		for (i,l) in enumerate(self.lines):
			l.set_ydata(self.yvals[:,i])
			l.set_xdata(self.xvals)
		plt.gca().set_xlim([max(0.,self.i-self.plotsize),max(self.i,self.plotsize)])
		plt.gca().relim()
		plt.gca().autoscale_view()
		tolegend = self.get_nbest(4,0.5)
		plt.legend(handles=[self.lines[i] for i in tolegend])
		plt.draw()
		plt.pause(0.0001)
	
	
	def get_nbest(self,n=0,th=0.0):
		order = np.argsort(-self.buffer)
		order = [i for i in order if self.buffer[i]>=th]
		if(n>0):
			return order[:n]
		else:
			return order
		
	
	def print_nbest(self,n=0,th=0.0):
		ordertoprint = self.get_nbest(n,th)
		print("Present products:")
		message = ""
		for i in ordertoprint:
			print("- %s (%s): %.3f" % (labels_to_tags[classes[i]],classes[i],self.buffer[i]))
			message += classes[i] + "|" #
		# message = classes[randint(0, 29)] + "|" + classes[randint(0, 29)]
		return message #
