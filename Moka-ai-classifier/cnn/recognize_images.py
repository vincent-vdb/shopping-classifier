#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alexandre Coninx
    ISIR CNRS/UPMC
    21/07/2016
""" 

import camerainterface

import sys

if __name__ == "__main__":
    print("\n")

    def usageexit():
        print("Usage: %s [images_dir]" % sys.argv[0])
        print("\n")
        sys.exit(1)
    if len(sys.argv) < 2:
        usageexit()

    imgdir = sys.argv[1].strip()
    plotting = True
    # print("hello")

    if len(sys.argv) > 2 and sys.argv[2] == 'false':
        plotting = False
    print("Input dir: %s" % imgdir)
    source = camerainterface.ImagesPlaybackSource(imgdir, loop=True)
    pi = camerainterface.ProductIdentifier(source, plotting)
    #pi.start()
    pi.start_vince_multi()
    sys.exit(0)
