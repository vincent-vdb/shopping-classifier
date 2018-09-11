#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 22:54:51 2017

@author: endy
"""
import numpy as np
import os

from matplotlib import cm

default_cmap = cm.gist_heat

from PIL import Image, ImageOps, ImageDraw, ImageFont
import classif_mehdi_fcn

fnt = ImageFont.truetype('FreeSans.ttf', 48)

import gen_ydata

thres_max=0.2

tags_to_labels, labels_to_tags = gen_ydata.load_mappings()


def hmap_to_img(hmap,cmap=default_cmap):
	hmapimage = Image.fromarray(np.uint8(cmap(hmap)*255))
	return hmapimage


def process_fname(fullname):
	elems = fullname.split('/')
	outdir='-'.join(elems[-3:]).split('.')[0]
	m = classif_mehdi_fcn.matcher.match(fullname)
	if(m):
		classes = classif_mehdi_fcn.decode_classes(m.groups())
	else:
		classes = list()
	return (outdir, classes)


def print_csv(hmaps,groundtruth,outfilecsv):
	maxes = np.max(hmaps,axis=(0,1))[:len(classif_mehdi_fcn.classes)]
	means = np.mean(hmaps,axis=(0,1))[:len(classif_mehdi_fcn.classes)]
	argmaxes = np.argsort(-maxes)
	argmeans = np.argsort(-means)
	with open(outfilecsv,'w') as fd:
		fd.write("class,max,max_rank,mean,mean_rank,in_gt\n")
		for i,c in enumerate(classif_mehdi_fcn.classes):
			fd.write("%s,%f,%d,%f,%d,%d\n" %(c,maxes[i],argmaxes[i],means[i],argmeans[i],(1 if c in groundtruth else 0)))


def get_allclasses(net,imgfile,outroot="outimages/"):
	origimage = Image.open(imgfile).convert('RGBA')
	hmaps = classif_mehdi_fcn.apply_to_image(net,imgfile)
	outdir, groundtruth = process_fname(imgfile)
	fulloutdir = "%s/%s" % (outroot,outdir)
	try:
		os.mkdir(fulloutdir)
	except OSError:
		print("WARNING:Output dir seems to already exist. It is probably a bad thing.")
	print_csv(hmaps,groundtruth,"%s/stats.csv" % fulloutdir)
	class_ok = list()
	for i,c in enumerate(classif_mehdi_fcn.classes):
		origimage.save("%s/original.jpg" % fulloutdir)
		if(np.max(hmaps[:,:,i]) > thres_max):
			smallmap = hmap_to_img(hmaps[:,:,i])
			imagemap = convert_map_format(smallmap,origimage.size)
			blended_img = Image.blend(origimage,imagemap,0.5)
			d = ImageDraw.Draw(blended_img)
			d.text((10,10),"%s (%s)" % (labels_to_tags[c],c),font=fnt,fill=(255,255,255,255))
			blended_img.save("%s/heatmap-%s.jpg" % (fulloutdir,c))
			class_ok.append(c)
	return class_ok




def convert_map_format(hmap,targetsize):
	smaller= np.min(targetsize)
	larger=np.max(targetsize)
	smaller_first = (np.argmin(targetsize) == 0)
	diff = larger - smaller
	hmap_ss = hmap.resize((smaller,smaller))
	hmap_bb = ImageOps.expand(hmap_ss,diff/2,fill=(0,0,0))
	if(smaller_first):
		hmap_final = hmap_bb.crop((diff/2,0,larger-diff/2,larger))
	else:
		hmap_final = hmap_bb.crop((0,diff/2,larger,larger-diff/2))
	return hmap_final