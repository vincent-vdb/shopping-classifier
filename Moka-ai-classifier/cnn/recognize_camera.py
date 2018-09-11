#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alexandre Coninx
    ISIR CNRS/UPMC
    21/07/2016
""" 

import numpy as np

import classif_mehdi_fcn, camerainterface

import sys

if __name__ == "__main__":
	print("\n")
	def usageexit():
		print("Usage: %s <video dev device> <plotting>" % sys.argv[0])
		print("\n")
		sys.exit(1)

	videodev = 0
	if(len(sys.argv) > 1):
		videodev = int(sys.argv[1])
		
	plotting = True
	if len(sys.argv) > 2 and sys.argv[2] == 'false':
		plotting = False
	print("Using video device %d" % videodev)
	source = camerainterface.CameraInterface(videodev)
	pi = camerainterface.ProductIdentifier(source, plotting)
	pi.start()
	sys.exit(0)
