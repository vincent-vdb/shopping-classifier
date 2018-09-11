#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alexandre Coninx
    ISIR CNRS/UPMC
    09/04/2017
""" 

from PIL import Image, ImageDraw, ImageFont, ImageOps

import numpy as np

import csv

mydrawcolor = (255,0,0) # red


def load_mappings(f="mappings.csv"):
	 with open(f, 'r') as fd:
		reader = csv.reader(fd, delimiter=',')
		tags_to_labels = dict()
		labels_to_tags = dict()
		for line in reader:
			classlabel, tag = line[:2]
			main = bool(int(line[2]))
			tags_to_labels[tag] = classlabel
			if(main):
				labels_to_tags[classlabel] = tag
		return tags_to_labels, labels_to_tags


def parse_csv(csvfile,nframes,mapping=None):
	if not mapping:
		mapping, _ = load_mappings()
	data =  [dict() for i in range(nframes)]
	fbad = open("badlabels.txt","a")
	fbad.close()
	#badlabels = list()
	with open(csvfile, 'r') as fd:
		reader = csv.reader(fd, delimiter=';')
		firstline = True
		for line in reader:
			#nom, si, sx, sy, sw, sh, sangle, skeyframe = line
			nom, si, sx, sy, sw, sh, sangle = line[:7] # don't care about keyframe
			if(firstline):
				nom = nom[3:]
				firstline = False
			nom = nom.upper()
			i,x,y,w,h = (int(x) for x in (si, sx, sy, sw, sh))
			angle = float(sangle.replace(',','.'))
			if(nom not in mapping.keys()):
				print("WARNING: unknown object %s" % nom)
				#badlabels.append(nom)
				continue
			if(i >= nframes):
				print("Frame %d is after the end, ignoring it (this is probably OK)" % i)
				continue
			data[i][mapping[nom]] = (x,y,w,h,angle)
	#with open("badlabels.txt","a") as fbad:
	#	for l in badlabels:
	#		fbad.write(l+"\n")
	return data
	
	
def rotate_point(px, py, cx,cy,alpha):
	x = px - cx
	y = py - cy
	xrot = np.cos(alpha)*x - np.sin(alpha)*y
	yrot = np.sin(alpha)*x + np.cos(alpha)*y
	return (xrot+cx, yrot+cy)
	
def compute_corners(x,y,w,h,angle):
	corners_norot = [(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
	cx, cy = x+w/2, y+h/2
	if(angle == 0.):
		return corners_norot, (cx, cy)
	else:
		return [rotate_point(p[0], p[1], cx, cy, angle) for p in corners_norot], (cx, cy)
		

def draw_box_on_img(box, img, center=None, label=None):
	draw = ImageDraw.Draw(img)
	draw.polygon(box,outline=mydrawcolor,fill=None)
	if(center and label):
		draw.text(center, label, fill=mydrawcolor)
	return img

def gen_img_with_box(box, basesize):
	im = Image.new('L',basesize)
	draw = ImageDraw.Draw(im)
	draw.polygon(box,outline=None,fill=255)
	return im


def get_image_with_tags(rootname,n,nmax=None):
	tags_to_labels, _ = load_mappings()
	if(not(nmax)):
		nmax = n+1
	csvfile = "%s.csv" % rootname
	imagefile = "%s/image-%04d.jpg" % (rootname, n+1)
	csvdata = parse_csv(csvfile,nmax,tags_to_labels)
	im = Image.open(imagefile)
	tagdata = csvdata[n]
	for k in tagdata.keys():
		bboxdata = tagdata[k]
		corners, center = compute_corners(*bboxdata)
		draw_box_on_img(corners, im, center=center, label=k)
	return im

def gen_ydata_from_csvdata(csvdata,classlist,basesize=(1080,1920), targetsize=(224,224), exclusive = False):
	nout = len(classlist) if not exclusive else len(classlist)+1
	yout = np.zeros(targetsize + (nout,))
	for (i,c) in enumerate(classlist):
		if(c not in csvdata.keys()):
			continue
		corners, _ = compute_corners(*csvdata[c])
		im = gen_img_with_box(corners, basesize)
		im = ImageOps.fit(im,targetsize,method=Image.ANTIALIAS,centering=(0.5,0.5))
		yout[:,:,i] = np.array(im,dtype=float)/255
	if exclusive:
		s = np.sum(yout,axis=(2))
		yout[:,:,-1][s==0.] = 1.
		s2 = np.sum(yout,axis=(2),keepdims=True)
		yout /= s2
	return yout


