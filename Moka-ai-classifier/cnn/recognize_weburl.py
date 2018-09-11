#!/usr/bin/python
# -*- coding: utf-8 -*-

import camerainterface

import sys

if __name__ == "__main__":
	print("\n")
	def usageexit():
		print("Usage: %s [weburl]" % sys.argv[0])
		print("\n")
		sys.exit(1)

	if(len(sys.argv) < 2):
		usageexit()

	imgUrl = sys.argv[1].strip()
	plotting = True
	if len(sys.argv) > 2 and sys.argv[2] == 'false':
		plotting = False
	print("Input imgUrl: %s" % imgUrl)
	source = camerainterface.ImagesPlaybackSource(imgUrl=imgUrl, loop=True)
	pi = camerainterface.ProductIdentifier(source, plotting)
	pi.start()
	sys.exit(0)
