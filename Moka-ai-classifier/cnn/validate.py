#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alexandre Coninx
    ISIR CNRS/UPMC
    03/06/2017
""" 

import numpy as np


import classif_mehdi_fcn

import os


def get_subset_ds(listfile,base_ds="data_mehdi/DATA/ds_12.txt"):
	biglist = classif_mehdi_fcn.gen_ds_list(base_ds)
	rmapping = {e[0]:i for (i,e) in enumerate(biglist)}
	smallist = list()
	with open(listfile,'r') as fd:
		for l in fd:
			videodir = l.strip()
			if videodir in rmapping.keys():
				smallist.append(biglist[rmapping[videodir]])
	return smallist

def master_ds(listfile,baseds="data_mehdi/DATA/ds_12.txt",fullds=["data_mehdi/DATA/ds_triple.txt","data_mehdi/DATA/ds_quad.txt"]):
	ds = get_subset_ds(listfile,baseds) 
	for l in fullds:
		ds += classif_mehdi_fcn.gen_ds_list(l)
	return ds


def simple_histo_criterion(histo,binthres=2,relthres=1e-3): #At least 1permille pixels >20%
	return (np.sum(histo[binthres:]) > relthres*np.sum(histo))

def compute_stats(pred,target):
	truepos = len(set(pred).intersection(set(target)))
	if(len(target)==0):
		recall = 1.
	else:
		recall = float(truepos)/len(target)
	
	if(len(pred)==0):
		precision = 1.
	else:
		precision = float(truepos)/len(pred)
	if(precision+recall==0):
		f1 = 0.
	else:
		f1 = 2*(recall*precision)/(recall+precision)
	return precision,recall,f1

def entropy_from_hmaps(hmaps):
	entropy = np.zeros(hmaps.shape[:2])
	for i in range(len(classif_mehdi_fcn.classes)):
		entropy += - hmaps[:,:,i]*np.log(hmaps[:,:,i])
	return entropy
		



def predict_ds(ds,net,outdir="outdata",entropy_th=1.0):
	try:
		os.mkdir(outdir)
	except OSError:
		print("WARNING:Output dir seems to already exist. It is probably a bad thing.")
	with open("%s/badreco.csv" % outdir,'w')as fdbad:
		fdbad.write("fname,pred,true,precision,recall,f1\n")
		for video in ds:
			dirname, n, y, trueclasses, _ = video
			print("Processing frames from %s" % dirname)
			localdirname=dirname.split("/")[-1]
			with open("%s/%s.csv" % (outdir,localdirname),'w') as fd:
				for i in range(n):
					classes_ok = list()
					imgfile = "%s/image-%04d.jpg" % (dirname,i+1) # Images start at 1
					hmaps = classif_mehdi_fcn.apply_to_image(net,imgfile)
					entropy = entropy_from_hmaps(hmaps)
					if(entropy_th):
						good_pixels = np.array(entropy<entropy_th,dtype=np.float)
					else:
						good_pixels = None
					fd.write("%d," % (i+1))
					for (i,c) in enumerate(classif_mehdi_fcn.classes):
						histos, _ = np.histogram(hmaps[:,:,i], range=(0.,1.), weights=good_pixels)
						if(simple_histo_criterion(histos)):
							classes_ok.append(c)
						for hbin in histos:
							fd.write("%d,"% hbin)
					fd.write("%s," % ("-".join(classes_ok)))
					precision,recall,f1 = compute_stats(classes_ok,trueclasses)
					fd.write("%.2f,%.2f,%.2f\n"%(precision,recall,f1))
					if(f1<1.):
						print imgfile
						fdbad.write("%s,%s,%s,%f,%f,%f\n" % (imgfile,("-".join(classes_ok)),("-".join(trueclasses)),precision,recall,f1))
	
			
				
					
							
						
				
			