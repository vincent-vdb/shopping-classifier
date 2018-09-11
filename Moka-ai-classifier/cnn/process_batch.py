#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alexandre Coninx
    ISIR CNRS/UPMC
    10/01/2017
""" 

import numpy as np

import classif_mehdi_multiclass_bigds as classif_mehdi

outdir="save-big-123"

list_all="data_mehdi/DATA/ds_1234.txt"
list_train="%s/train-ds-dump.txt" % outdir
list_valid="%s/valid-ds-dump.txt" % outdir

resultsdir="%s/stats" % outdir

defweights="weights-big-last.h5"


eval_batch_size = 64

def generate_data():
	with open(list_train,'r') as fd:
		trainlist = [line.strip() for line in fd]
	with open(list_valid,'r') as fd:
		validlist = [line.strip() for line in fd]
	dataset = list()
	with open(list_all,'r') as fd:
		for line in fd:
			[sample_dir_path, n_imgs] = line.strip().split('\t')
			m = classif_mehdi.matcher.match(sample_dir_path)
			if(m):
				g = m.groups()
				matching_classes = classif_mehdi.decode_classes(g)
				#y = classif_mehdi.classes_to_y(matching_classes)
				fullfpath = "%s/%s" % (classif_mehdi.dsdir, sample_dir_path)
				in_train = fullfpath in trainlist
				in_valid = fullfpath in validlist
				dataset.append((fullfpath,int(n_imgs),matching_classes,in_train,in_valid,sample_dir_path))
			else:
				print("WARNING: dir %s does not match pattern" % fpath)
	return dataset

def load_video(fullpath,n_imgs):
	return np.array([classif_mehdi.load_and_process_img("%s/image-%04d.jpg" % (fullpath,i)) for i in range(1,n_imgs+1)]).transpose((0,3,1,2))


def process_video(dataelement,model):
	x = load_video(dataelement[0],dataelement[1])
	y = model.predict(x,batch_size=eval_batch_size)
	return y
	
	
def process_all():
	ds = generate_data()
	model = classif_mehdi.load_network(defweights)
	for (videonumber,video) in enumerate(ds):
		if(videonumber<2922):
			continue
		with open("%s/predictions-%s.csv" % (resultsdir,video[5].replace("/","-")),"w") as fd_log:
			fd_log.write("dir,in_train,in_valid,n_imgs,correct_classes\n")
			fd_log.write("%s,%d,%d,%d,%s\n" % (video[5], video[3], video[4], video[1], "-".join(sorted(video[2]))))
			print("Processing video %s (%d/%d)" % (video[5],videonumber+1,len(ds)))
			y = process_video(video,model)
			print("Done")
			fd_log.write("frame,%s\n" % ",".join(classif_mehdi.classes))
			for i in range(video[1]):
				fd_log.write("%d" % (i+1))
				for c in range(len(classif_mehdi.classes)):
					fd_log.write(",%.3e" % y[i,c])
				fd_log.write("\n")

