#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alexandre Coninx
    ISIR CNRS/UPMC
    21/07/2016
""" 

import numpy as np

import classif_mehdi_fcn

import sys

#TODO fix me FCN conversion
if __name__ == "__main__":
	print("\n")
	def usageexit():
		print("Usage: %s [image_file]" % sys.argv[0])
		print("\n")
		sys.exit(1)

	if(len(sys.argv) < 2):
		usageexit()

	imgfile = sys.argv[1].strip()
	print("Input image: %s" % imgfile)
	classif_mehdi_fcn.eval_image(imgfile)
	sys.exit(0)
